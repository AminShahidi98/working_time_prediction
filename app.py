"""
Remaining working time of a work order - web app.

Run on your own computer with:
    streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import (CHECKPOINTS, FEATURES, History, build_live_row, predict_range,
                       train_models)

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

# The images are loaded straight from GitHub: on Streamlit Cloud the server
# answered 500 when it served the local files itself.
ASSETS = 'https://raw.githubusercontent.com/AminShahidi98/working_time_prediction/main/assets/'

# GBTwin palette: the green of the logo, with Quindi greys
MODEL_GREEN = '#0A926F'
GREEN_TEXT = '#076B51'      # the same green, dark enough to read as text on white
PLAN_GREY = '#526860'
REAL_RED = '#C0392B'
SPEED_BLUE = '#2E6DA4'
BAND = 'rgba(10, 146, 111, 0.18)'

st.set_page_config(page_title='Remaining working time · Quindi',
                   page_icon=ASSETS + 'gbtwin_icon.png', layout='wide')
st.logo(ASSETS + 'gbtwin_logo.png', size='large',
        icon_image=ASSETS + 'gbtwin_icon.png')


# =====================================================================
# loading (once per server start)
# =====================================================================
@st.cache_resource(show_spinner='Loading the data and training the three models...')
def load_everything():
    with open(os.path.join(ARTIFACTS, 'meta.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    orders_all = pd.read_parquet(os.path.join(ARTIFACTS, 'orders_all.parquet'))
    orders_planned = pd.read_parquet(os.path.join(ARTIFACTS, 'orders_planned.parquet'))
    alarms = pd.read_parquet(os.path.join(ARTIFACTS, 'alarms.parquet'))
    train = pd.read_parquet(os.path.join(ARTIFACTS, 'train.parquet'))
    test_rows = pd.read_parquet(os.path.join(ARTIFACTS, 'test_rows.parquet'))
    test_curves = pd.read_parquet(os.path.join(ARTIFACTS, 'test_curves.parquet'))

    history = History(orders_all, orders_planned, alarms)
    models = train_models(train)

    low, middle, high = predict_range(models, test_rows)
    test_rows = test_rows.copy()
    test_rows['ml_p10_h'] = low
    test_rows['ml_pred_h'] = middle
    test_rows['ml_p90_h'] = high

    # the largest orders the model ever learned from, to warn about extrapolation
    limits = {
        'ordered_qty': float(train['ordered_qty'].max()),
        'plan_work_h': float(train['plan_work_h'].max()),
        'plan_cycle_s': float(train['plan_cycle_s'].max()),
    }

    return {
        'meta': meta,
        'limits': limits,
        'orders_all': orders_all,
        'orders_planned': orders_planned,
        'history': history,
        'models': models,
        'test_rows': test_rows,
        'test_curves': test_curves,
    }


data = load_everything()
meta = data['meta']


def machine_label(wc_id):
    info = meta['machines'].get(str(int(wc_id)))
    if info is None:
        return 'machine %d' % int(wc_id)
    return '%d - %s' % (int(wc_id), info['name'])


def hours_text(hours):
    whole = int(hours)
    minutes = int(round((hours - whole) * 60))
    if minutes == 60:
        whole = whole + 1
        minutes = 0
    return '%.2f h  (%d h %02d min)' % (hours, whole, minutes)


# =====================================================================
# sidebar
# =====================================================================
with st.sidebar:
    st.markdown('### Remaining working time')
    st.caption('A machine-learning correction to the MES plan, built on MES and '
               'SCADA data. Master thesis, University of Padua.')
    st.divider()
    cost_per_hour = st.number_input('Cost of one machine hour (€)', min_value=0.0,
                                    value=65.0, step=1.0,
                                    help='Used only to express hours in euros. Set 0 to hide.')
    st.divider()
    result = meta['result']
    st.markdown('**On %d test predictions it never saw**' % result['test_rows'])
    st.markdown('- error **%.1f%% → %.1f%%** of the hours left' % (
        result['mes_error_pct'], result['ml_error_pct']))
    st.markdown('- closer than the plan on **%.1f%%** of them' % result['win_rate'])
    st.caption('Data from %s to %s. Model trained on orders started before %s.' % (
        meta['data_first_day'], meta['data_last_day'], meta['split_date']))


# =====================================================================
# shared pieces of the forms
# =====================================================================
def article_table_for(machine_id):
    """Articles made on this machine, most frequent first (as in notebook cell 35)."""
    made_here = data['orders_all'][data['orders_all']['wc_id'] == machine_id]
    rows = []
    for article_code, group in made_here.groupby('ar_code'):
        total_pieces = group['goods'].sum()
        if total_pieces <= 0:
            continue
        rows.append({'ar_code': article_code, 'orders': len(group),
                     'median_qty': float(group['goods'].median())})
    rows.sort(key=lambda item: -item['orders'])
    return rows, made_here


def order_form(prefix):
    """The order itself: machine, article, quantity and plan. Returns a dict."""
    studied = meta['studied_machines']
    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        machine_id = st.selectbox(
            'Machine', studied, format_func=machine_label, key=prefix + '_machine',
            help='The workcenter that runs the order. Only the %d machines with at least 5 '
                 'orders in the test period can be chosen, because the model was trained and '
                 'checked on them.' % len(studied))

    article_rows, made_here = article_table_for(machine_id)
    article_options = []
    for item in article_rows:
        article_options.append(item['ar_code'])
    article_options.append('NEW')

    def article_label(code):
        if code == 'NEW':
            return 'a new article (never made on this machine)'
        for item in article_rows:
            if item['ar_code'] == code:
                return '%s   (%d past orders)' % (code, item['orders'])
        return code

    with column_2:
        article_code = st.selectbox(
            'Article', article_options, format_func=article_label,
            key=prefix + '_article_%d' % machine_id,
            help='The product being made. The list shows the articles this machine has '
                 'produced, most frequent first. The model uses the past orders of the article: '
                 'how fast it really ran and how far from the plan it usually was. Choose '
                 '"a new article" for one the machine has never made; the model then relies on '
                 'the plan and the history of the machine only.')

    # --- defaults ---------------------------------------------------------
    # As in notebook cell 35, with one change: when the article has no
    # planned orders, the setup time comes from the machine's planned
    # orders instead of 0, because every planned order in the data has a
    # setup time between 0.5 and 8 hours and the model never saw 0.
    planned = data['orders_planned']
    planned_on_machine = planned[planned['wc_id'] == machine_id]

    if article_code == 'NEW':
        default_qty = float(made_here['goods'].median())
        history_article = 'NEW-ARTICLE'
        qty_source = 'the median quantity of all past orders on this machine'
        planned_before = planned_on_machine
        plan_source = '%d planned orders of other articles on this machine' % len(planned_before)
    else:
        default_qty = 1.0
        for item in article_rows:
            if item['ar_code'] == article_code:
                default_qty = item['median_qty']
        history_article = article_code
        qty_source = 'the median quantity of the past orders of this article'
        planned_before = planned[planned['ar_code'] == article_code]
        plan_source = '%d past orders of this article that had an MES plan' % len(planned_before)

    if len(planned_before) > 0:
        default_cycle = float(planned_before['plan_cycle_s'].median())
        cycle_source = 'the median of the %s' % plan_source
    else:
        default_cycle = float(made_here['working_seconds'].sum() / max(made_here['goods'].sum(), 1))
        cycle_source = ('this article has no past order with an MES plan, so the default is its '
                        'REAL past speed on this machine. Replace it with the plan if you have it')

    if len(planned_before) > 0:
        setup_frame = planned_before
        setup_source = 'the median of the %s' % plan_source
    else:
        setup_frame = planned_on_machine
        setup_source = ('this article has no past order with an MES plan, so the default is the '
                        'median of the %d planned orders on this machine' % len(planned_on_machine))
    if len(setup_frame) > 0:
        default_setup = float(setup_frame['wo_expected_setup_time'].median() / 3600.0)
    else:
        default_setup = 0.0

    state = prefix + '_%d_%s' % (machine_id, article_code)
    with column_3:
        ordered_qty = st.number_input(
            'Ordered quantity (pieces)', min_value=1.0,
            value=float(round(max(default_qty, 1))), step=100.0, format='%.0f',
            key=state + '_qty',
            help='The number of pieces ordered, as written in the MES before production '
                 'starts. The checkpoints and the plan are both measured on this quantity. '
                 'Default: %s.' % qty_source)

    column_1, column_2, column_3, column_4 = st.columns(4)
    with column_1:
        plan_cycle_s = st.number_input(
            'MES plan: seconds per piece', min_value=0.01,
            value=round(default_cycle, 2), step=0.1, format='%.2f', key=state + '_cycle',
            help='The cycle time the MES plan assumes for one piece. The plan for the whole '
                 'order is this times the ordered quantity, and the model corrects it. '
                 'Default: %s.' % cycle_source)
    with column_2:
        plan_setup_h = st.number_input(
            'Planned setup (hours)', min_value=0.0,
            value=round(default_setup, 2), step=0.25, format='%.2f', key=state + '_setup',
            help='The time the MES plans for preparing the machine before production: mould '
                 'change, adjustments, first-piece checks. The model reads it as information '
                 'about the order, but setup is NOT part of the predicted hours, which are '
                 'working time only. Default: %s.' % setup_source)
    with column_3:
        priority = st.selectbox(
            'Priority', [0, 1], key=state + '_priority',
            help='The priority flag of the order as written in the MES plan. In the data, 155 '
                 'of the 220 planned orders have 0 and 65 have 1.')
    with column_4:
        last_day = pd.Timestamp(meta['data_last_day'])
        start_day = st.date_input(
            'Order started on', value=last_day.date(), key=state + '_day',
            help='The day the first piece was (or will be) made. The model uses the history of '
                 'the article and the machine up to this day, and the alarms in the 7 and 30 '
                 'days before it. Default: the last day in the data, %s.' % meta['data_last_day'])
        start_time = st.time_input(
            'at', value=pd.Timestamp('06:00').time(), key=state + '_time',
            help='The time the order started. Together with the clock hours that have passed, '
                 'it gives the day of the week and the hour of the prediction, which the model '
                 'also uses.')

    limits = data['limits']
    plan_hours = ordered_qty * plan_cycle_s / 3600.0
    if (ordered_qty > limits['ordered_qty'] or plan_hours > limits['plan_work_h']
            or plan_cycle_s > limits['plan_cycle_s']):
        st.warning('This order is larger than any order the model learned from (up to %s pieces '
                   'and %.0f planned hours). The answer below is an extrapolation and should not '
                   'be trusted.' % (format(int(limits['ordered_qty']), ','), limits['plan_work_h']))

    start = pd.Timestamp.combine(start_day, start_time)
    if start > last_day + pd.Timedelta(days=1):
        st.info('The data ends on %s. The history of this article and machine is taken as '
                'it stood then, and no alarms after that day are known.' % meta['data_last_day'])

    return {
        'machine_id': machine_id,
        'article_code': history_article,
        'article_shown': article_label(article_code),
        'ordered_qty': ordered_qty,
        'plan_cycle_s': plan_cycle_s,
        'plan_setup_h': plan_setup_h,
        'priority': priority,
        'start': start,
        'state': state,
        'typical_wall_frac': meta['machines'][str(machine_id)]['typical_wall_frac'],
    }


def finish_lines(figure, ends):
    """A dotted vertical line at each finishing time, with its value near the axis."""
    heights = [0.03, 0.11, 0.19, 0.27]
    for index in range(len(ends)):
        value, colour, dash = ends[index]
        figure.add_vline(x=value, line_color=colour, line_dash=dash,
                         line_width=1, opacity=0.7)
        text_colour = colour
        if colour == MODEL_GREEN:
            text_colour = GREEN_TEXT
        figure.add_annotation(x=value, y=heights[index % len(heights)], yref='paper',
                              text=' %.2f h' % value, showarrow=False, xanchor='left',
                              font=dict(color=text_colour, size=12))


def pieces_chart(title, start_x, start_y, total_pieces, plan_end, model_end,
                 low_end, high_end, past_x, past_y, past_label, past_colour,
                 extra_end=None, real_end=None):
    figure = go.Figure()

    figure.add_trace(go.Scatter(
        x=[start_x, low_end, high_end, start_x], y=[start_y, total_pieces, total_pieces, start_y],
        mode='lines', fill='toself', fillcolor=BAND, line=dict(width=0), hoverinfo='skip',
        name='model range p10 – p90'))
    figure.add_trace(go.Scatter(
        x=past_x, y=past_y, mode='lines', line=dict(color=past_colour, width=2.5),
        name=past_label))
    figure.add_trace(go.Scatter(
        x=[start_x, plan_end], y=[start_y, total_pieces], mode='lines+markers',
        line=dict(color=PLAN_GREY, width=2, dash='dash'), marker=dict(size=[0, 9]),
        name='MES plan'))
    figure.add_trace(go.Scatter(
        x=[start_x, model_end], y=[start_y, total_pieces], mode='lines+markers',
        line=dict(color=MODEL_GREEN, width=3), marker=dict(size=[0, 10]),
        name='model (most likely)'))
    if extra_end is not None:
        x_value, colour, label = extra_end
        figure.add_trace(go.Scatter(
            x=[start_x, x_value], y=[start_y, total_pieces], mode='lines+markers',
            line=dict(color=colour, width=2, dash='dot'), marker=dict(size=[0, 8]),
            name=label))
    figure.add_trace(go.Scatter(
        x=[start_x], y=[start_y], mode='markers',
        marker=dict(symbol='star', size=18, color='gold', line=dict(color='black', width=1)),
        name='now'))

    ends = [(plan_end, PLAN_GREY, 'dash'), (model_end, MODEL_GREEN, 'solid')]
    if extra_end is not None:
        ends.append((extra_end[0], extra_end[1], 'dot'))
    if real_end is not None:
        ends.append((real_end, REAL_RED, 'solid'))
    finish_lines(figure, ends)

    widest = max([plan_end, model_end, high_end] + list(past_x))
    figure.update_layout(
        title=title, height=520, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title='machine working hours since the order started',
                   range=[0, widest * 1.12], gridcolor='#E4E7EB'),
        yaxis=dict(title='pieces produced', range=[0, total_pieces * 1.06],
                   gridcolor='#E4E7EB', tickformat=','),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='left', x=0),
        plot_bgcolor='white', hovermode='closest')
    return figure


def history_expander(row):
    with st.expander('What the model knew about this machine and article'):
        column_1, column_2, column_3 = st.columns(3)
        with column_1:
            st.markdown('**Article history**')
            st.write('past orders: %d' % row['art_n_past'])
            if pd.notna(row['art_cycle_s']):
                st.write('real seconds per piece: %.2f' % row['art_cycle_s'])
            if pd.notna(row['art_plan_bias']):
                st.write('real / planned time: %.2f' % row['art_plan_bias'])
        with column_2:
            st.markdown('**Machine history**')
            st.write('past orders: %d' % row['mach_n_past'])
            if pd.notna(row['mach_cycle_s']):
                st.write('real seconds per piece: %.2f' % row['mach_cycle_s'])
            if pd.notna(row['mach_plan_bias']):
                st.write('real / planned time: %.2f' % row['mach_plan_bias'])
        with column_3:
            st.markdown('**Alarms before the start**')
            st.write('last 7 days: %d alarms, %.1f h' % (row['alarms_7d'], row['alarm_h_7d']))
            st.write('last 30 days: %d alarms, %.1f h' % (row['alarms_30d'], row['alarm_h_30d']))
        st.caption('A value above 1 for real / planned time means past orders needed more '
                   'time than the MES planned.')


# =====================================================================
# the page
# =====================================================================
st.title('How much working time does this order still need?')

tab_running, tab_plan, tab_real, tab_model, tab_about = st.tabs([
    'Order in progress', 'Plan a new order', 'Real orders (test period)', 'Inside the model',
    'How it works'])


# ---------------------------------------------------------------------
# tab 1: an order that is running now
# ---------------------------------------------------------------------
with tab_running:
    st.markdown('Describe the order and how far it has got. The model corrects the MES plan '
                'using the progress so far and the history of the article and the machine.')
    order = order_form('run')

    st.markdown('##### How far the order has got')
    planned_speed_h = order['ordered_qty'] * order['plan_cycle_s'] / 3600.0
    column_1, column_2, column_3, column_4 = st.columns(4)
    with column_1:
        pieces_done = st.number_input(
            'Pieces produced so far', min_value=0.0, max_value=float(order['ordered_qty']),
            value=float(round(order['ordered_qty'] * 0.4)), step=100.0, format='%.0f',
            key=order['state'] + '_done_%d' % order['ordered_qty'],
            help='Good pieces made so far, as SCADA counts them. Enter 0 if the order has not '
                 'started: the model then predicts from the plan and the history only, as at '
                 'the 0% checkpoint. Default: 40% of the ordered quantity, as an example.')
    with column_2:
        worked_h = st.number_input(
            'Machine working hours so far', min_value=0.0,
            value=round(pieces_done * order['plan_cycle_s'] / 3600.0, 2), step=0.25,
            format='%.2f', key=order['state'] + '_worked_%d' % pieces_done,
            help='Hours the machine has actually spent producing this order, as SCADA records '
                 'them: no stops, nights or setup. Together with the pieces it gives the speed '
                 'the order is really running at, which is the strongest signal the model has. '
                 'Default: the hours the plan would need for the pieces above, that is, an '
                 'order running exactly on plan.')
    with column_3:
        typical_elapsed = worked_h / order['typical_wall_frac'] if worked_h > 0 else 0.0
        elapsed_h = st.number_input(
            'Clock hours since the start', min_value=0.0,
            value=round(typical_elapsed, 2), step=1.0, format='%.2f',
            key=order['state'] + '_elapsed_%.2f' % worked_h,
            help='Real time passed since the order started, including nights, weekends and '
                 'stops. It tells the model how much of the clock the machine spends producing, '
                 'and when the prediction is made. Default: estimated from how much of the clock '
                 'this machine usually spends producing (%.0f%%).'
                 % (order['typical_wall_frac'] * 100))
    with column_4:
        wastes_done = st.number_input(
            'Waste pieces so far', min_value=0.0, value=0.0, step=1.0, format='%.0f',
            key=order['state'] + '_waste',
            help='Pieces scrapped so far. In the plant\'s records waste was always zero, so the '
                 'model never learned anything from it and this value does not change the '
                 'prediction. It is kept so the inputs match the model\'s.')

    if pieces_done > 0 and worked_h <= 0:
        st.warning('Pieces were produced, so the working hours so far must be above zero.')
        st.stop()
    if elapsed_h > 0 and elapsed_h < worked_h:
        st.warning('Clock hours cannot be fewer than working hours; using the working hours instead.')
        elapsed_h = worked_h
    if pieces_done == 0:
        st.caption('Nothing produced yet: the prediction is made as at the 0% checkpoint, '
                   'from the plan and the history only.')

    row = build_live_row(data['history'], order['machine_id'], order['article_code'],
                         order['ordered_qty'], order['plan_cycle_s'], order['plan_setup_h'],
                         order['priority'], order['start'], pieces_done, worked_h,
                         elapsed_h, wastes_done, order['typical_wall_frac'])
    low, middle, high = predict_range(data['models'], row)
    p10, p50, p90 = float(low[0]), float(middle[0]), float(high[0])
    plan_left = float(row['mes_remaining_h'].iloc[0])
    worked_now = float(row['worked_h'].iloc[0])
    correction = p50 / max(plan_left, 0.05)

    st.divider()
    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric('Hours still needed (most likely)', '%.2f h' % p50,
                    delta='%+.2f h vs the plan' % (p50 - plan_left), delta_color='inverse')
    column_2.metric('Range p10 – p90 (hours)', '%.1f – %.1f' % (p10, p90),
                    help='%.2f – %.2f h' % (p10, p90))
    column_3.metric('The MES plan says', '%.2f h' % plan_left)
    column_4.metric('Correction to the plan', '× %.2f' % correction)

    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric('Whole order, model', '%.2f h' % (worked_now + p50),
                    help='Hours already worked + hours predicted to be left.')
    column_2.metric('Whole order, MES plan', '%.2f h' % (worked_now + plan_left))
    if cost_per_hour > 0:
        column_3.metric('Remaining cost, model', '€ %s' % format(round(p50 * cost_per_hour), ','),
                        help='Range: € %s – %s' % (format(round(p10 * cost_per_hour), ','),
                                                   format(round(p90 * cost_per_hour), ',')))
        column_4.metric('Remaining cost, MES plan', '€ %s' % format(round(plan_left * cost_per_hour), ','))

    if row['remaining_pieces'].iloc[0] == 0:
        st.info('All the ordered pieces are produced, so the plan expects nothing more. '
                'The model may still expect a little extra time, as real orders often overrun.')

    figure = pieces_chart(
        title='%s on machine %s' % (order['article_shown'], machine_label(order['machine_id'])),
        start_x=worked_now, start_y=pieces_done, total_pieces=order['ordered_qty'],
        plan_end=worked_now + plan_left, model_end=worked_now + p50,
        low_end=worked_now + p10, high_end=worked_now + p90,
        past_x=[0.0, worked_now], past_y=[0.0, pieces_done],
        past_label='produced so far', past_colour='#7B8794')
    st.plotly_chart(figure)
    st.caption('The grey line is a straight summary of the progress entered above. The shaded '
               'wedge is the range the model expects in about 8 orders out of 10.')

    history_expander(row.iloc[0])


# ---------------------------------------------------------------------
# tab 2: plan a new order (notebook cell 35)
# ---------------------------------------------------------------------
with tab_plan:
    st.markdown('Describe an order that has **not started yet**. Before the first piece nobody knows '
                'how fast it will run, so the answer below uses only what is known in advance: the '
                'plan, and how this article and this machine really ran in the past.')
    plan_order = order_form('plan')

    # --- the answer before production starts: no speed needed ----------
    start_row = build_live_row(data['history'], plan_order['machine_id'],
                               plan_order['article_code'], plan_order['ordered_qty'],
                               plan_order['plan_cycle_s'], plan_order['plan_setup_h'],
                               plan_order['priority'], plan_order['start'], 0.0, 0.0,
                               None, 0.0, plan_order['typical_wall_frac'])
    low, middle, high = predict_range(data['models'], start_row)
    p10, p50, p90 = float(low[0]), float(middle[0]), float(high[0])
    plan_whole = float(start_row['mes_remaining_h'].iloc[0])

    st.divider()
    st.markdown('##### Before the first piece')
    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric('Working hours the order will need (most likely)', '%.2f h' % p50,
                    delta='%+.2f h vs the plan' % (p50 - plan_whole), delta_color='inverse')
    column_2.metric('Range p10 – p90 (hours)', '%.1f – %.1f' % (p10, p90),
                    help='%.2f – %.2f h. The real working time should fall inside this range in '
                         'about 8 orders out of 10.' % (p10, p90))
    column_3.metric('The MES plan says', '%.2f h' % plan_whole)
    column_4.metric('Correction to the plan', '× %.2f' % (p50 / max(plan_whole, 0.05)))
    if cost_per_hour > 0:
        column_1, column_2, column_3, column_4 = st.columns(4)
        column_1.metric('Machine cost, model', '€ %s' % format(round(p50 * cost_per_hour), ','),
                        help='Range: € %s – %s' % (format(round(p10 * cost_per_hour), ','),
                                                   format(round(p90 * cost_per_hour), ',')))
        column_2.metric('Machine cost, MES plan', '€ %s' % format(round(plan_whole * cost_per_hour), ','))
    history_expander(start_row.iloc[0])

    # --- what if: the speed is a scenario, not an input -----------------
    st.divider()
    st.markdown('##### What if the order runs faster or slower than planned?')

    planned = data['orders_planned']
    article_bias = start_row['art_plan_bias'].iloc[0]
    machine_bias = start_row['mach_plan_bias'].iloc[0]
    if pd.notna(article_bias):
        typical_speed = float(article_bias)
        past_planned = planned[(planned['ar_code'] == plan_order['article_code']) &
                               (planned['finish'] < plan_order['start'])]
        speed_source = ('this article: its past %d orders with a plan needed × %.2f the planned '
                        'time, in the median' % (len(past_planned), typical_speed))
    elif pd.notna(machine_bias):
        typical_speed = float(machine_bias)
        speed_source = ('this machine, because the article has no planned history: the machine\'s '
                        'past orders needed × %.2f the planned time, in the median' % typical_speed)
    else:
        typical_speed = 1.0
        speed_source = 'the plan itself, because no history is available'
    slider_default = round(min(max(typical_speed, 0.6), 1.8), 2)

    st.markdown('The speed only becomes known once the order runs. Here you can try a scenario: the '
                'app imagines the order running at the speed you choose and asks the model what it '
                'would say at each later checkpoint. The starting value is the speed typical for '
                '%s.' % speed_source)

    speed = st.slider('Scenario: time per piece compared with the plan  '
                      '(1.0 = as planned, 1.2 = 20% slower, 0.9 = 10% faster)',
                      min_value=0.6, max_value=1.8, value=float(slider_default), step=0.01,
                      key=plan_order['state'] + '_speed',
                      help='Not something you need to know in advance. It is a what-if: each piece '
                           'is assumed to take this many times the planned seconds per piece, for '
                           'the whole order. The blue dotted line in the chart is what that speed '
                           'alone would give; the green line is what the model would answer. '
                           'Starting value: the speed typical for %s.' % speed_source)

    rows = []
    for checkpoint in CHECKPOINTS:
        pieces = checkpoint * plan_order['ordered_qty']
        hours = pieces * plan_order['plan_cycle_s'] * speed / 3600.0
        rows.append(build_live_row(data['history'], plan_order['machine_id'],
                                   plan_order['article_code'], plan_order['ordered_qty'],
                                   plan_order['plan_cycle_s'], plan_order['plan_setup_h'],
                                   plan_order['priority'], plan_order['start'], pieces, hours,
                                   None, 0.0, plan_order['typical_wall_frac']))
    simulated = pd.concat(rows, ignore_index=True)
    low, middle, high = predict_range(data['models'], simulated)
    simulated['p10'] = low
    simulated['p50'] = middle
    simulated['p90'] = high
    simulated['same_speed'] = (simulated['remaining_pieces'] * plan_order['plan_cycle_s']
                               * speed / 3600.0)

    labels = []
    percents = []
    for checkpoint in CHECKPOINTS:
        labels.append('%d%%' % round(checkpoint * 100))
        percents.append(round(checkpoint * 100))

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=percents, y=simulated['p90'], mode='lines',
                                line=dict(width=0), showlegend=False, hoverinfo='skip'))
    figure.add_trace(go.Scatter(x=percents, y=simulated['p10'], mode='lines', fill='tonexty',
                                fillcolor=BAND, line=dict(width=0), name='model range p10 – p90'))
    figure.add_trace(go.Scatter(x=percents, y=simulated['mes_remaining_h'], mode='lines+markers',
                                line=dict(color=PLAN_GREY, dash='dash', width=2), name='MES plan'))
    figure.add_trace(go.Scatter(x=percents, y=simulated['same_speed'], mode='lines+markers',
                                line=dict(color=SPEED_BLUE, dash='dot', width=2),
                                name='if the scenario speed holds'))
    figure.add_trace(go.Scatter(x=percents, y=simulated['p50'], mode='lines+markers',
                                line=dict(color=MODEL_GREEN, width=3), name='model (most likely)'))
    figure.update_layout(
        title='Working hours still to come, at each checkpoint, in this scenario', height=470,
        margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white',
        xaxis=dict(title='share of the ordered pieces already produced', gridcolor='#E4E7EB',
                   ticksuffix='%', dtick=10),
        yaxis=dict(title='hours still to come', gridcolor='#E4E7EB', rangemode='tozero'),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='left', x=0))
    st.plotly_chart(figure)
    st.caption('At 0% the model lines do not depend on the scenario, because nothing has been '
               'produced yet. From 10% on, the model sees the scenario speed as if it had been '
               'measured.')

    table = pd.DataFrame({
        'checkpoint': labels,
        'pieces made': simulated['pieces_done'].round(0).astype(int),
        'plan left (h)': simulated['mes_remaining_h'].round(2),
        'model left (h)': simulated['p50'].round(2),
        'p10 (h)': simulated['p10'].round(2),
        'p90 (h)': simulated['p90'].round(2),
        'correction': (simulated['p50'] / simulated['mes_remaining_h'].clip(lower=0.05)).round(2),
    })
    st.dataframe(table, hide_index=True)


# ---------------------------------------------------------------------
# tab 3: real orders from the test period
# ---------------------------------------------------------------------
with tab_real:
    st.markdown('Real orders the model **never saw while learning**: all of them started after %s. '
                'Pick one and a moment inside it, and compare the plan and the model with what '
                'really happened.' % meta['split_date'])
    test_rows = data['test_rows']

    column_1, column_2, column_3 = st.columns([1, 1.4, 1.6])
    with column_1:
        real_machine = st.selectbox('Machine', meta['studied_machines'],
                                    format_func=machine_label, key='real_machine',
                                    help='One of the machines the model was tested on.')
    on_machine = test_rows[test_rows['wc_id'] == real_machine]
    order_list = []
    for code, group in on_machine.groupby('wo_code'):
        order_list.append((group['start'].iloc[0], code))
    order_list.sort()
    order_codes = []
    for item in order_list:
        order_codes.append(item[1])

    def order_label(code):
        first_row = on_machine[on_machine['wo_code'] == code].iloc[0]
        return '%s  ·  %s  ·  %s pcs  ·  %s' % (
            code, first_row['ar_code'], format(int(first_row['ordered_qty']), ','),
            first_row['start'].strftime('%d %b %Y'))

    with column_2:
        real_code = st.selectbox('Order', order_codes, format_func=order_label, key='real_order',
                                 help='Orders of this machine from the test period, oldest '
                                      'first: order code, article, ordered pieces, start date.')
    this_order = on_machine[on_machine['wo_code'] == real_code].sort_values('checkpoint')
    available = []
    for value in this_order['checkpoint']:
        available.append('%d%%' % round(value * 100))
    with column_3:
        chosen_label = st.select_slider('Moment inside the order (share of ordered pieces made)',
                                        options=available, value=available[min(3, len(available) - 1)],
                                        key='real_checkpoint_%s' % real_code,
                                        help='The share of the ordered pieces already made when '
                                             'the prediction is made. The model only sees the order '
                                             'up to this point. Checkpoints the order never reached '
                                             'are not offered.')
    chosen = this_order.iloc[available.index(chosen_label)]

    curve = data['test_curves'][data['test_curves']['wo_code'] == real_code]
    total_pieces = float(curve['pieces'].iloc[-1])
    real_end = float(curve['hours'].iloc[-1])
    start_x = float(chosen['worked_h'])
    start_y = float(chosen['pieces_done'])

    plan_left = float(chosen['mes_remaining_h'])
    p50 = float(chosen['ml_pred_h'])
    p10 = float(chosen['ml_p10_h'])
    p90 = float(chosen['ml_p90_h'])
    real_left = float(chosen['target_h'])

    denominator = max(real_left, 0.5)
    plan_pct = (plan_left - real_left) / denominator * 100
    model_pct = (p50 - real_left) / denominator * 100

    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric('Really left', '%.2f h' % real_left)
    column_2.metric('MES plan said', '%.2f h' % plan_left, delta='%+.1f%% off' % plan_pct,
                    delta_color='off')
    column_3.metric('Model said', '%.2f h' % p50, delta='%+.1f%% off' % model_pct,
                    delta_color='off')
    if p10 <= real_left <= p90:
        verdict = 'yes'
    else:
        verdict = 'no'
    column_4.metric('Real value inside p10 – p90?', verdict,
                    help='range %.2f – %.2f h' % (p10, p90))

    if abs(model_pct) < abs(plan_pct):
        st.success('At this moment the model is closer to the truth than the plan.')
    elif abs(model_pct) > abs(plan_pct):
        st.warning('At this moment the plan is closer to the truth than the model.')

    figure = pieces_chart(
        title='Order %s on machine %s' % (real_code, machine_label(real_machine)),
        start_x=start_x, start_y=start_y, total_pieces=total_pieces,
        plan_end=start_x + plan_left, model_end=start_x + p50,
        low_end=start_x + p10, high_end=start_x + p90,
        past_x=list(curve['hours']), past_y=list(curve['pieces']),
        past_label='what really happened', past_colour=REAL_RED,
        real_end=real_end)
    st.plotly_chart(figure)

    every_moment = pd.DataFrame({
        'checkpoint': available,
        'really left (h)': this_order['target_h'].round(2).values,
        'plan said (h)': this_order['mes_remaining_h'].round(2).values,
        'model said (h)': this_order['ml_pred_h'].round(2).values,
        'p10 – p90 (h)': [('%.2f – %.2f' % (a, b)) for a, b in
                          zip(this_order['ml_p10_h'], this_order['ml_p90_h'])],
    })
    st.markdown('##### The same order at every checkpoint')
    st.dataframe(every_moment, hide_index=True)
    st.caption('As in the thesis, the pieces still to make are counted from the quantity the '
               'order finally produced, for both the plan and the model.')


# ---------------------------------------------------------------------
# tab 4: inside the model
# ---------------------------------------------------------------------
FEATURE_WORDS = {
    'ordered_qty': 'pieces ordered', 'plan_work_h': 'planned working hours',
    'plan_cycle_s': 'planned seconds per piece', 'plan_setup_h': 'planned setup hours',
    'priority': 'order priority', 'pct_of_ordered': 'share of ordered pieces made',
    'pieces_done': 'pieces made so far', 'remaining_pieces': 'pieces still to make',
    'worked_h': 'machine hours so far', 'obs_cycle_s': 'seconds per piece so far',
    'elapsed_wall_h': 'clock hours since the start',
    'wall_work_frac': 'share of the clock spent working', 'waste_rate': 'waste per piece',
    'dow': 'day of the week', 'hour': 'hour of the day',
    'art_n_past': 'past orders of the article', 'art_cycle_s': 'article: past seconds per piece',
    'art_med_work_h': 'article: typical order hours', 'art_plan_bias': 'article: past real ÷ plan',
    'mach_n_past': 'past orders on the machine', 'mach_cycle_s': 'machine: past seconds per piece',
    'mach_med_work_h': 'machine: typical order hours', 'mach_plan_bias': 'machine: past real ÷ plan',
    'alarms_7d': 'alarms, last 7 days', 'alarm_h_7d': 'alarm hours, last 7 days',
    'alarms_30d': 'alarms, last 30 days', 'alarm_h_30d': 'alarm hours, last 30 days',
    'obs_vs_plan': 'time per piece so far ÷ plan',
    'obs_vs_art': 'time per piece so far ÷ article\'s past',
    'obs_vs_mach': 'time per piece so far ÷ machine\'s past',
    'plan_vs_art': 'planned time per piece ÷ article\'s past',
    'plan_vs_mach': 'planned time per piece ÷ machine\'s past',
    'est_from_obs_h': 'hours left at the speed so far',
    'est_from_art_h': 'hours left at the article\'s past speed',
    'est_from_mach_h': 'hours left at the machine\'s past speed',
    'est_plan_x_mach_bias': 'plan × machine\'s past real ÷ plan',
    'est_plan_x_art_bias': 'plan × article\'s past real ÷ plan',
    'mes_remaining_h': 'plan: hours left', 'wc_id': 'machine',
}


@st.cache_resource(show_spinner='Opening up the trees...')
def boosting_internals():
    """What the p50 model is made of, read from the fitted model.

    These are scikit-learn internals (version pinned in requirements.txt):
    the trees are in _predictors, the starting value in _baseline_prediction
    and the column order the trees use in _preprocessor.
    """
    model = data['models']['p50']
    test_rows = data['test_rows']
    X = test_rows[FEATURES]

    # the trees see the categorical column (the machine) first, then the others
    tree_columns = []
    for feature, is_category in zip(FEATURES, model.is_categorical_):
        if is_category:
            tree_columns.append(feature)
    for feature, is_category in zip(FEATURES, model.is_categorical_):
        if not is_category:
            tree_columns.append(feature)

    trees = [tree[0] for tree in model._predictors]

    # how many times each input is used as a question, in all the trees
    questions = {}
    for tree in trees:
        for node in tree.nodes:
            if not node['is_leaf']:
                feature = tree_columns[node['feature_idx']]
                questions[feature] = questions.get(feature, 0) + 1

    return {
        'model': model,
        'trees': trees,
        'start': float(model._baseline_prediction.ravel()[0]),
        'tree_columns': tree_columns,
        'machines': list(model._preprocessor.named_transformers_['encoder'].categories_[0]),
        'encoded': model._preprocessor.transform(X),
        # the log-correction after 1, 2, ... all trees, for every test row
        'staged': np.array(list(model.staged_predict(X))),
        'questions': questions,
        'can_be_empty': set(feature for feature in FEATURES if X[feature].isna().any()),
    }


def tree_path(tree, values):
    """The nodes one row visits in one tree, from the root to its leaf."""
    nodes = tree.nodes
    node = 0
    path = [0]
    while not nodes[node]['is_leaf']:
        split = nodes[node]
        value = values[split['feature_idx']]
        if np.isnan(value):
            go_left = bool(split['missing_go_to_left'])
        elif split['is_categorical']:
            bits = tree.raw_left_cat_bitsets[split['bitset_idx']]
            code = int(value)
            go_left = bool((bits[code // 32] >> (code % 32)) & 1)
        else:
            go_left = value <= split['num_threshold']
        if go_left:
            node = int(split['left'])
        else:
            node = int(split['right'])
        path.append(node)
    return path


def leaves_below(nodes, index):
    """The leaves under one node of a tree."""
    if nodes[index]['is_leaf']:
        return [index]
    return (leaves_below(nodes, int(nodes[index]['left'])) +
            leaves_below(nodes, int(nodes[index]['right'])))


def tree_picture(tree, internals, path, whole_tree):
    """One tree as a Graphviz drawing, with the path of one row in green.

    Unless whole_tree is set, every branch the row does not take is folded
    into one grey box, so the drawing stays small enough to read.
    """
    nodes = tree.nodes
    on_path = set(path)
    # the folded view is a narrow column, so it reads best top to bottom
    direction = 'LR' if whole_tree else 'TB'
    lines = ['digraph {', 'rankdir=%s; bgcolor="transparent"; nodesep=0.3; ranksep=0.35;' % direction,
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
             'margin="0.14,0.06"];', 'edge [fontname="Helvetica", fontsize=10, color="#9AA5B1"];']
    for index in range(len(nodes)):
        node = nodes[index]
        if not whole_tree and index not in on_path:
            parent_on_path = False
            for parent in path:
                if not nodes[parent]['is_leaf'] and index in (nodes[parent]['left'],
                                                               nodes[parent]['right']):
                    parent_on_path = True
            if not parent_on_path:
                continue
            if not node['is_leaf']:
                folded = leaves_below(nodes, index)
                factors = np.exp(nodes[folded]['value'])
                lines.append('n%d [label="%d more leaves\\n× %.3f … × %.3f\\n%d rows", '
                             'fillcolor="#FFFFFF", color="#C8CDD3", fontcolor="#7B8794", '
                             'style="rounded,filled,dashed"];'
                             % (index, len(folded), factors.min(), factors.max(), node['count']))
                continue
        if node['is_leaf']:
            factor = np.exp(node['value'])
            fill = '#FDEBD3' if factor >= 1 else '#DCE9F5'
            label = '× %.3f\\n%d rows' % (factor, node['count'])
            border = '#076B51' if index in on_path else '#9AA5B1'
            width = 3 if index in on_path else 1
            lines.append('n%d [label="%s", fillcolor="%s", color="%s", penwidth=%d];'
                         % (index, label, fill, border, width))
            continue
        feature = internals['tree_columns'][node['feature_idx']]
        if node['is_categorical']:
            bits = tree.raw_left_cat_bitsets[node['bitset_idx']]
            left_machines = []
            for code in range(len(internals['machines'])):
                if (bits[code // 32] >> (code % 32)) & 1:
                    left_machines.append(str(int(internals['machines'][code])))
            question = 'machine one of\\n%s ?' % ', '.join(left_machines)
        else:
            question = '%s\\n≤ %s ?' % (FEATURE_WORDS.get(feature, feature),
                                        '%.3g' % node['num_threshold'])
        label = '%s\\n%d rows' % (question, node['count'])
        if index in on_path:
            lines.append('n%d [label="%s", fillcolor="#0A926F", fontcolor="white", '
                         'color="#076B51"];' % (index, label))
        else:
            lines.append('n%d [label="%s", fillcolor="#F4F5F9", color="#9AA5B1"];'
                         % (index, label))
        yes_label = 'yes'
        no_label = 'no'
        if feature in internals['can_be_empty']:
            if node['missing_go_to_left']:
                yes_label = 'yes / empty'
            else:
                no_label = 'no / empty'
        for child, text in [(node['left'], yes_label), (node['right'], no_label)]:
            if not whole_tree and index not in on_path:
                continue
            if index in on_path and child in on_path:
                style = 'color="#0A926F", penwidth=3, fontcolor="#076B51"'
            else:
                style = ''
            lines.append('n%d -> n%d [label=" %s", %s];' % (index, child, text, style))
    lines.append('}')
    return '\n'.join(lines)


with tab_model:
    internals = boosting_internals()
    model = internals['model']
    test_rows = data['test_rows']
    n_trees = len(internals['trees'])

    st.markdown('### The model: Gradient Boosting Decision Trees (GBDT)')
    st.markdown(
        'The model is a **gradient-boosted decision tree regressor**: scikit-learn\'s '
        '`HistGradientBoostingRegressor`. It is a chain of **%d small decision trees**, built one '
        'after the other, where each new tree learns to fix the mistakes the trees before it '
        'still make. This is what *gradient boosting* means. This tab opens the trees up and '
        'follows one real test order through them.' % n_trees)

    # --- the order followed through the whole tab -------------------------------
    at_30 = test_rows[test_rows['checkpoint'] == 0.3].sort_values('target_h')
    example_codes = list(at_30['wo_code'])
    # the order shown in the thesis presentation: the plan said 5.18 h, the model 8.03 h
    first_example = len(example_codes) // 2
    if 'WO-0578' in example_codes:
        first_example = example_codes.index('WO-0578')
    column_1, column_2 = st.columns(2)
    with column_1:
        example_code = st.selectbox(
            'Order to follow (a test order, when 30% of its pieces were made)', example_codes,
            index=first_example, key='tree_example',
            format_func=lambda code: '%s  ·  %s' % (
                code, machine_label(at_30[at_30['wo_code'] == code]['wc_id'].iloc[0])),
            help='Any test order that reached 30% of its ordered pieces. The model never saw it '
                 'while learning.')
    position = int(np.flatnonzero((test_rows['wo_code'] == example_code).values &
                                  (test_rows['checkpoint'] == 0.3).values)[0])
    example = test_rows.iloc[position]
    curve = np.concatenate([[internals['start']], internals['staged'][:, position]])
    steps = np.diff(curve)
    plan_left = float(example['mes_remaining_h'])
    factor = float(np.exp(curve[-1]))
    truth = float(np.log(max(example['target_h'], 0.05) / max(plan_left, 0.05)))

    # --- 1. the idea --------------------------------------------------------------
    st.markdown('#### 1 · The idea: the trees correct the plan')
    st.graphviz_chart('''
digraph {
    rankdir=LR; bgcolor="transparent"; nodesep=0.25;
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
          color="#9AA5B1", fillcolor="#F4F5F9", margin="0.14,0.07"];
    edge [color="#7B8794"];
    inputs [label="what is known\\nat this moment\\n(39 inputs)"];
    start  [label="start\\n%+.3f"];
    t1     [label="tree 1\\n%+.4f", fillcolor="#E3F4EF", color="#0A926F"];
    t2     [label="tree 2\\n%+.4f", fillcolor="#E3F4EF", color="#0A926F"];
    more   [label="…", shape=plaintext, style=""];
    tn     [label="tree %d\\n%+.4f", fillcolor="#E3F4EF", color="#0A926F"];
    total  [label="sum %+.3f\\ncorrection × %.2f", fillcolor="#0A926F", fontcolor="white",
            color="#076B51"];
    plan   [label="MES plan\\n%.2f h left"];
    answer [label="%.2f h × %.2f\\n= %.2f h left", fillcolor="#232523", fontcolor="white",
            color="#232523"];
    inputs -> t1 [style=dashed]; inputs -> t2 [style=dashed]; inputs -> tn [style=dashed];
    start -> t1 -> t2 -> more -> tn -> total -> answer;
    plan -> answer;
}
''' % (internals['start'], steps[0], steps[1], n_trees, steps[-1], curve[-1], factor,
       plan_left, plan_left, factor, plan_left * factor))
    st.markdown(
        '- The trees never predict hours. Together they predict **how wrong the plan is**.\n'
        '- Each tree looks at the inputs and adds one small number. Their sum becomes a '
        'correction factor: × 1.00 would leave the plan unchanged, × %.2f stretches it.\n'
        '- The answer is **plan × correction**. For order %s that is %.2f h; in reality '
        '%.2f h were needed.'
        % (factor, example_code, plan_left * factor, example['target_h']))
    st.caption('The p10 – p90 range comes from two more chains of trees, built the same way but '
               'aiming low and high. Everything on this tab is the middle one (p50).')

    with st.expander('Why a correction, and why the numbers are logarithms'):
        st.markdown(
            'What the trees learn for every training row is '
            '**ln( real hours left ÷ plan hours left )**: 0 means the plan was right, +0.18 '
            'means 20% more time (e⁰·¹⁸ = × 1.20), −0.11 means 10% less (× 0.90).\n\n'
            'Trees can only **add** their numbers, but slowdowns **multiply**: two slowdowns of '
            '10% are 1.1 × 1.1 = 1.21, not 1.1 + 1.1 = 2.2. In logarithms, adding is '
            'multiplying: ln 1.1 + ln 1.1 = 0.19, and e⁰·¹⁹ = 1.21. So trees that add can model '
            'slowdowns that multiply.\n\n'
            'And a model that learned nothing would add up to 0, giving × 1.00: the plan, '
            'unchanged. The plan is always the starting point.')

    # --- 2. one tree --------------------------------------------------------------
    st.markdown('#### 2 · One tree: a few yes/no questions')
    column_1, column_2 = st.columns(2)
    with column_2:
        tree_number = st.slider('Tree to look at', min_value=1, max_value=n_trees, value=1,
                                key='tree_number',
                                help='Tree 1 is the first one built; each later tree fixes what '
                                     'the earlier ones still got wrong.')
    tree = internals['trees'][tree_number - 1]
    path = tree_path(tree, internals['encoded'][position])
    st.graphviz_chart(tree_picture(tree, internals, path, whole_tree=False))
    leaf = tree.nodes[path[-1]]
    st.markdown(
        'Order **%s** answers %d questions (the green path) and ends in a **leaf** that adds '
        '**%+.4f**, that is × %.3f. Orange leaves stretch the plan, blue leaves shrink it. '
        'Dashed boxes fold up the branches this order does not take; the counts are the training '
        'rows that reached each box. *empty* shows where a row goes when a value does not exist '
        'yet, for example the speed so far at 0%%.'
        % (example_code, len(path) - 1, leaf['value'], np.exp(leaf['value'])))
    with st.expander('Show the whole tree (%d leaves)' % tree.get_n_leaf_nodes()):
        st.graphviz_chart(tree_picture(tree, internals, path, whole_tree=True))
        st.caption('Use the full-screen button at the top right of the drawing to read it.')
    st.caption('Each tree is at most %d questions deep and every leaf rests on at least %d '
               'training rows, so no tree can learn a single order by heart. Leaves are small '
               'because each tree\'s answer is multiplied by the learning rate (%.2f): no single '
               'tree decides.' % (model.max_depth, model.min_samples_leaf, model.learning_rate))

    # --- 3. the trees add up --------------------------------------------------------
    st.markdown('#### 3 · Many trees: each one fixes what is left')
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(range(len(curve))), y=np.exp(curve), mode='lines',
                                line=dict(color=MODEL_GREEN, width=3), name='model after n trees'))
    figure.add_hline(y=np.exp(truth), line_color=REAL_RED, line_dash='dash')
    figure.add_annotation(x=1, xref='paper', y=np.exp(truth), text='the truth: × %.2f' % np.exp(truth),
                          showarrow=False, yshift=10, xanchor='right', font=dict(color=REAL_RED))
    figure.add_hline(y=1, line_color=PLAN_GREY, line_dash='dot')
    figure.add_annotation(x=0, xref='paper', y=1, text='the plan: × 1.00', showarrow=False,
                          yshift=-12, xanchor='left', font=dict(color=PLAN_GREY))
    figure.update_layout(
        title='Order %s: the correction after 0, 1, 2 … %d trees' % (example_code, n_trees),
        height=360, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white', showlegend=False,
        xaxis=dict(title='number of trees added', gridcolor='#E4E7EB'),
        yaxis=dict(title='correction to the plan (×)', gridcolor='#E4E7EB'))
    st.plotly_chart(figure)
    st.markdown(
        'This is **gradient boosting**. It starts from one guess for every order (× %.2f, the '
        'typical correction in the training data). Tree 1 is trained on the mistakes of that '
        'guess, tree 2 on the mistakes that are still left, and so on. Adding the trees one by '
        'one moves the correction step by step; for one single order it does not have to land '
        'exactly on the truth.' % np.exp(internals['start']))

    with st.expander('See the first trees one by one'):
        shown = 5
        names = ['start'] + ['tree %d' % (k + 1) for k in range(shown)]
        names += ['trees %d – %d' % (shown + 1, len(steps)), 'sum']
        values = ([internals['start']] + list(steps[:shown]) +
                  [float(curve[-1] - curve[shown]), curve[-1]])
        figure = go.Figure(go.Waterfall(
            x=names, y=values, measure=['absolute'] + ['relative'] * (shown + 1) + ['total'],
            text=['%+.3f' % value for value in values], textposition='outside',
            increasing=dict(marker=dict(color='#E08A1E')),
            decreasing=dict(marker=dict(color=SPEED_BLUE)),
            totals=dict(marker=dict(color=MODEL_GREEN)), connector=dict(line=dict(color='#9AA5B1'))))
        figure.add_hline(y=truth, line_color=REAL_RED, line_dash='dash')
        figure.add_annotation(x=1, xref='paper', y=truth, text='the truth: %+.3f' % truth,
                              showarrow=False, yshift=10, xanchor='right', font=dict(color=REAL_RED))
        figure.update_layout(
            height=380, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='white',
            showlegend=False, yaxis=dict(title='sum so far (log units)', gridcolor='#E4E7EB'))
        st.plotly_chart(figure)
        st.caption('Each bar is the leaf the order lands in, in one tree. Orange adds, blue '
                   'subtracts. The correction is e to the power of the sum: e^%.3f = × %.2f.'
                   % (curve[-1], factor))

    # --- 4. when to stop ------------------------------------------------------------
    st.markdown('#### 4 · When to stop adding trees')
    train_loss = -np.array(model.train_score_)
    held_back_loss = -np.array(model.validation_score_)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(range(len(train_loss))), y=train_loss, mode='lines',
                                line=dict(color=SPEED_BLUE, width=2), name='rows it learns from'))
    figure.add_trace(go.Scatter(x=list(range(len(held_back_loss))), y=held_back_loss,
                                mode='lines', line=dict(color=MODEL_GREEN, width=3),
                                name='rows kept aside to check (%d%%)'
                                     % round(model.validation_fraction * 100)))
    figure.add_vline(x=model.n_iter_, line_color='#232523', line_dash='dash')
    figure.add_annotation(x=model.n_iter_, y=1, yref='paper', text='stop: %d trees ' % model.n_iter_,
                          showarrow=False, xanchor='right')
    figure.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='white',
        xaxis=dict(title='number of trees', gridcolor='#E4E7EB'),
        yaxis=dict(title='average error (log units)', gridcolor='#E4E7EB'),
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='left', x=0))
    st.plotly_chart(figure)
    st.markdown(
        'Too many trees start fixing random noise instead of real patterns. So %d%% of the '
        'training rows are kept aside: after every tree the model checks its error on them, and '
        'once %d trees in a row bring no real gain, it stops. It was allowed up to %d trees and '
        'stopped at **%d**.'
        % (round(model.validation_fraction * 100), model.n_iter_no_change, model.max_iter,
           model.n_iter_))

    # --- 5. what the trees ask about ---------------------------------------------------
    st.markdown('#### 5 · What the trees ask about most')
    asked = sorted(internals['questions'].items(), key=lambda item: item[1])[-12:]
    labels = []
    counts = []
    for feature, count in asked:
        labels.append('%s  (%s)' % (FEATURE_WORDS.get(feature, feature), feature))
        counts.append(count)
    figure = go.Figure(go.Bar(x=counts, y=labels, orientation='h',
                              marker=dict(color=MODEL_GREEN), text=counts, textposition='outside'))
    figure.update_layout(
        height=420, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='white',
        xaxis=dict(title='questions asked, in all %d trees' % n_trees, gridcolor='#E4E7EB',
                   range=[0, max(counts) * 1.2]))
    st.plotly_chart(figure)
    st.caption('The 12 inputs the trees ask about most often. An input asked about often is one '
               'the model finds useful for splitting the orders into groups that behave alike.')

    # --- 6. why trees ------------------------------------------------------------------
    st.markdown('#### 6 · Why trees suit this problem')
    st.markdown(
        '- **Empty values are fine.** At 0%% the speed so far does not exist yet; each question '
        'learns which way to send an empty value, instead of us inventing one.\n'
        '- **Units do not matter.** Pieces are in thousands, cycle times in seconds; a tree only '
        'asks "above or below this value?".\n'
        '- **Works with little data.** %s training rows is far too few for a neural network, '
        'but normal for boosted trees.\n'
        '- **Handles categories.** The machine is a name, not a quantity; one question can group '
        'any set of machines.\n'
        '- **Not thrown off by extreme orders.** The trees are trained on the absolute error, so '
        'the few orders that ran several times slower than planned do not dominate.'
        % format(meta['training_rows'], ','))
    with st.expander('One tree, a random forest, or gradient boosting?'):
        st.dataframe(pd.DataFrame({
            '': ['trees', 'how they are built', 'each tree learns', 'combined by',
                 'size of each tree', 'defence against overfitting'],
            'one decision tree': ['one', '–', 'the answer', '–', 'as deep as needed',
                                  'limit its depth'],
            'random forest': ['many', 'independently, side by side',
                              'the whole answer, from random rows and features', 'average',
                              'usually deep', 'randomness, then averaging'],
            'gradient boosting (this model)': ['%d' % n_trees, 'one after the other',
                                               'only the mistake left so far', 'adding them up',
                                               'small (depth %d)' % model.max_depth,
                                               'learning rate and early stopping'],
        }), hide_index=True)


# ---------------------------------------------------------------------
# tab 5: how it works
# ---------------------------------------------------------------------
with tab_about:
    result = meta['result']
    st.markdown('#### The idea')
    st.markdown(
        'The MES gives every order one expected working time, fixed before the first piece is '
        'made and never updated. SCADA records what the machines really do. The model reads both '
        'and predicts **how wrong the plan is** for this order, as a correction factor. A model '
        'that learns nothing returns × 1.00, which is the plan unchanged.')
    st.markdown(
        'Three gradient-boosted tree models are used, identical except for their loss: the first '
        'returns the most likely value (p50), the other two a lower and an upper bound (p10 and '
        'p90), so every prediction comes with a range.')

    st.markdown('#### The result on %d test predictions (%d orders, %d machines)' % (
        result['test_rows'], result['test_orders'], len(meta['studied_machines'])))
    summary = pd.DataFrame({
        'measure': ['average error (hours)', 'average percentage error',
                    'median percentage error', 'average bias (hours)'],
        'MES plan': ['%.2f' % result['mes_error_h'], '%.1f%%' % result['mes_error_pct'],
                     '%.1f%%' % result['mes_median_pct'], '%+.2f' % result['mes_bias_h']],
        'model': ['%.2f' % result['ml_error_h'], '%.1f%%' % result['ml_error_pct'],
                  '%.1f%%' % result['ml_median_pct'], '%+.2f' % result['ml_bias_h']],
    })
    st.dataframe(summary, hide_index=True)
    st.markdown('The model is closer to the real remaining time than the plan on **%.1f%%** of the '
                'test predictions. The real value falls inside the p10 – p90 range in **%.1f%%** of '
                'them (the target is 80%%).' % (result['win_rate'], result['range_coverage']))

    st.markdown('#### Good to know')
    st.markdown(
        '- Predictions are in **machine working hours**, not calendar time: nights, weekends and '
        'stops are not included.\n'
        '- Only the %d machines with enough test orders can be chosen.\n'
        '- In the first two tabs the pieces still to make are counted from the **ordered** '
        'quantity, because the final quantity of a running order is not known yet.\n'
        '- The history of an article or machine is taken as it stood when the order started, '
        'using data up to %s.' % (len(meta['studied_machines']), meta['data_last_day']))
