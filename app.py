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

from sklearn.inspection import permutation_importance

from predictor import (CHECKPOINTS, FEATURES, History, build_live_row, predict_range,
                       train_models)

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

# The images are loaded straight from GitHub: on Streamlit Cloud the server
# answered 500 when it served the local files itself.
ASSETS = 'https://raw.githubusercontent.com/AminShahidi98/working_time_prediction/main/assets/'

# Quindi palette (QuindiTheme in ProRob.ProNetManager.Edge)
MODEL_GREEN = '#04CF88'
GREEN_TEXT = '#037A51'      # the same green, dark enough to read as text on white
PLAN_GREY = '#526860'
REAL_RED = '#C0392B'
SPEED_BLUE = '#2E6DA4'
BAND = 'rgba(4, 207, 136, 0.18)'

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
FEATURE_GROUPS = [
    ('The MES plan', '#526860',
     ['ordered_qty', 'plan_work_h', 'plan_cycle_s', 'plan_setup_h', 'priority',
      'mes_remaining_h']),
    ('Progress so far', '#04CF88',
     ['pct_of_ordered', 'pieces_done', 'remaining_pieces', 'worked_h', 'obs_cycle_s',
      'elapsed_wall_h', 'wall_work_frac', 'waste_rate']),
    ('Day and hour', '#9AA5B1', ['dow', 'hour']),
    ('Article history', '#E08A1E',
     ['art_n_past', 'art_cycle_s', 'art_med_work_h', 'art_plan_bias']),
    ('Machine history', '#2E6DA4',
     ['mach_n_past', 'mach_cycle_s', 'mach_med_work_h', 'mach_plan_bias', 'wc_id']),
    ('Alarms', '#C0392B', ['alarms_7d', 'alarm_h_7d', 'alarms_30d', 'alarm_h_30d']),
    ('Comparisons and ready-made estimates', '#8E5BB5',
     ['obs_vs_plan', 'obs_vs_art', 'obs_vs_mach', 'plan_vs_art', 'plan_vs_mach',
      'est_from_obs_h', 'est_from_art_h', 'est_from_mach_h',
      'est_plan_x_mach_bias', 'est_plan_x_art_bias']),
]


def group_of(feature):
    for name, colour, features in FEATURE_GROUPS:
        if feature in features:
            return name, colour
    return 'other', '#9AA5B1'


@st.cache_resource(show_spinner='Measuring which inputs matter most...')
def feature_importance():
    """Permutation importance of the p50 model on the test rows.

    Each input is shuffled in turn; the more the error grows, the more the
    model relies on that input.
    """
    test_rows = data['test_rows']
    plan = test_rows['mes_remaining_h'].clip(lower=0.05)
    real = test_rows['target_h'].clip(lower=0.05)
    target = np.log(real / plan)
    model = data['models']['p50']
    baseline = float(np.mean(np.abs(model.predict(test_rows[FEATURES]) - target)))
    shuffled = permutation_importance(model, test_rows[FEATURES], target,
                                      scoring='neg_mean_absolute_error',
                                      n_repeats=5, random_state=42)
    rows = []
    for index, feature in enumerate(FEATURES):
        group, colour = group_of(feature)
        rows.append({'feature': feature, 'group': group, 'colour': colour,
                     'growth_pct': shuffled.importances_mean[index] / baseline * 100})
    return pd.DataFrame(rows).sort_values('growth_pct', ascending=False)


with tab_model:
    result = meta['result']
    test_rows = data['test_rows']

    st.markdown('This tab opens the model up: what goes in, what comes out, how it learned, '
                'and how well it does on orders it never saw.')

    # --- 1. the big picture -------------------------------------------------
    st.markdown('#### 1 · From factory data to a range of hours')
    st.graphviz_chart('''
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
          color="#526860", fillcolor="#F4F5F9", margin="0.15,0.08"];
    edge [color="#7B8794"];

    mes    [label="MES\\nwork orders\\n(the plan)"];
    scada  [label="SCADA\\nproduction records\\n(what really ran)"];
    alarms [label="SCADA\\nmachine alarms"];

    features [label="39 inputs at a checkpoint\\nplan · progress so far ·\\narticle & machine history ·\\nalarms · ratios",
              fillcolor="#E6FAF3", color="#04CF88"];

    p10 [label="model p10\\n(low edge)",  fillcolor="#FFFFFF"];
    p50 [label="model p50\\n(most likely)", fillcolor="#04CF88", fontcolor="white", color="#037A51"];
    p90 [label="model p90\\n(high edge)", fillcolor="#FFFFFF"];

    factor [label="correction factor\\n× c", shape=ellipse, fillcolor="#FFFFFF"];
    answer [label="hours still to come\\n= plan × c\\nwith a p10 – p90 range",
            fillcolor="#232523", fontcolor="white", color="#232523"];

    mes -> features; scada -> features; alarms -> features;
    features -> p10; features -> p50; features -> p90;
    p10 -> factor; p50 -> factor; p90 -> factor;
    factor -> answer;
}
''')
    st.markdown(
        '- At a **checkpoint** (0%, 10%, … 90% of the ordered pieces made) the app collects '
        'everything known *at that moment*, never anything from the future.\n'
        '- Three tree models read those inputs. They do not predict hours directly: they predict '
        '**how wrong the plan is**, as a correction factor *c*.\n'
        '- The answer is the plan\'s remaining hours times *c*. A factor of × 1.00 means the plan '
        'is right; × 1.20 means 20% more time than planned.')

    # --- 2. why a correction is needed --------------------------------------
    st.markdown('#### 2 · The plan is usually too optimistic')
    planned = data['orders_planned']
    real_cycle = planned['working_seconds'] / planned['goods'].clip(lower=1)
    plan_ratio = real_cycle / planned['plan_cycle_s']
    share_slower = float((plan_ratio > 1).mean() * 100)

    figure = go.Figure()
    figure.add_trace(go.Histogram(x=plan_ratio.clip(upper=3), xbins=dict(start=0, end=3, size=0.1),
                                  marker=dict(color=PLAN_GREY, line=dict(color='white', width=1)),
                                  name='orders'))
    figure.add_vline(x=1, line_color=MODEL_GREEN, line_width=3)
    figure.add_annotation(x=1, y=1, yref='paper', text=' the plan is exactly right', showarrow=False,
                          xanchor='left', font=dict(color=GREEN_TEXT))
    figure.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='white', showlegend=False,
        bargap=0.05,
        xaxis=dict(title='real time per piece ÷ planned time per piece  (values above 3 shown at 3)',
                   gridcolor='#E4E7EB'),
        yaxis=dict(title='number of orders', gridcolor='#E4E7EB'))
    st.plotly_chart(figure)
    st.caption('All %d completed orders that had an MES plan. %.0f%% of them needed more time per '
               'piece than planned (right of the green line). This steady gap is what the model '
               'learns to correct.' % (len(planned), share_slower))

    # --- 3. how it learned ------------------------------------------------
    st.markdown('#### 3 · Learning from the past, tested on the future')
    split_date = pd.Timestamp(meta['split_date'])
    starts = data['orders_all']['start']
    week = 7 * 24 * 3600 * 1000
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=starts[starts < split_date], xbins=dict(size=week),
                                  marker=dict(color=SPEED_BLUE), name='used to learn'))
    figure.add_trace(go.Histogram(x=starts[starts >= split_date], xbins=dict(size=week),
                                  marker=dict(color=MODEL_GREEN), name='kept hidden for the test'))
    figure.add_vline(x=split_date, line_color='#232523', line_dash='dash')
    figure.add_annotation(x=split_date, y=1, yref='paper', text=' split: %s' % meta['split_date'],
                          showarrow=False, xanchor='left')
    figure.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='white', barmode='overlay',
        xaxis=dict(title='week the order started', gridcolor='#E4E7EB'),
        yaxis=dict(title='orders started', gridcolor='#E4E7EB'),
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='left', x=0))
    st.plotly_chart(figure)
    st.markdown(
        'The orders are split **by date**, not at random: the model learns from the older orders '
        'and is tested only on orders that started after %s. This is how it would be used for '
        'real, always predicting the future from the past. Every training order is turned into '
        'one row per checkpoint, which gives %s training rows.'
        % (meta['split_date'], format(meta['training_rows'], ',')))

    column_1, column_2 = st.columns(2)
    with column_1:
        st.markdown('**What one model is**')
        st.markdown(
            'A *gradient-boosted* model is a long chain of small decision trees. Each tree asks a '
            'few yes/no questions, such as *"is the order running slower than planned?"* or '
            '*"did this article overrun in the past?"*, and adds a small correction. Each new tree '
            'fixes part of the mistakes the earlier ones still make.')
    with column_2:
        st.markdown('**Why three models**')
        st.markdown(
            'The three models are identical except for what they aim at. **p50** aims at the '
            'middle: half of the real values should be above it, half below. **p10** aims low, so '
            'only 1 real value in 10 is below it. **p90** aims high, so only 1 in 10 is above. '
            'Together they give the most likely value and a range.')

    # --- 4. one prediction, step by step -------------------------------------
    st.markdown('#### 4 · One prediction, step by step')
    at_30 = test_rows[test_rows['checkpoint'] == 0.3].sort_values('target_h').reset_index(drop=True)
    example = at_30.iloc[len(at_30) // 2]
    factor = example['ml_pred_h'] / max(example['mes_remaining_h'], 0.05)
    st.markdown('A real test order, **%s** on machine **%s**, when 30%% of its pieces were made:'
                % (example['wo_code'], machine_label(example['wc_id'])))
    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric('1 · The plan says', '%.2f h' % example['mes_remaining_h'])
    if pd.notna(example['obs_vs_plan']):
        column_2.metric('2 · Speed so far vs plan', '× %.2f' % example['obs_vs_plan'],
                        help='Real seconds per piece so far ÷ planned seconds per piece.')
    column_3.metric('3 · Model\'s correction', '× %.2f' % factor)
    column_4.metric('4 · Answer', '%.2f h' % example['ml_pred_h'],
                    help='Range p10 – p90: %.2f – %.2f h' % (example['ml_p10_h'], example['ml_p90_h']))
    st.markdown('%.2f h × %.2f = **%.2f h**, with a range of %.2f – %.2f h. What really happened: '
                '**%.2f h** were still needed.' % (
                    example['mes_remaining_h'], factor, example['ml_pred_h'],
                    example['ml_p10_h'], example['ml_p90_h'], example['target_h']))

    # --- 5. what the model looks at -------------------------------------------
    st.markdown('#### 5 · What the model looks at')
    importance = feature_importance()
    top = importance.head(12).iloc[::-1]
    figure = go.Figure()
    for name, colour, features in FEATURE_GROUPS:
        part = top[top['group'] == name]
        if len(part) == 0:
            continue
        figure.add_trace(go.Bar(x=part['growth_pct'], y=part['feature'], orientation='h',
                                marker=dict(color=colour), name=name))
    figure.update_layout(
        height=440, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='white', barmode='stack',
        xaxis=dict(title='how much the error grows when this input is scrambled (%)',
                   gridcolor='#E4E7EB'),
        yaxis=dict(categoryorder='array', categoryarray=list(top['feature'])),
        legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='left', x=0))
    st.plotly_chart(figure)
    st.caption('The 12 most useful of the 39 inputs, for the p50 model on the test orders. Each '
               'input is shuffled in turn, breaking its link with the truth; the more the error '
               'grows, the more the model relies on it.')

    with st.expander('All 39 inputs, by group'):
        for name, colour, features in FEATURE_GROUPS:
            st.markdown('<span style="color:%s">■</span> **%s**: %s'
                        % (colour, name, ', '.join('`%s`' % f for f in features)),
                        unsafe_allow_html=True)

    # --- 6. how well it does --------------------------------------------------
    st.markdown('#### 6 · How well it does on orders it never saw')
    column_1, column_2 = st.columns(2)
    # zoomed on 98% of the points, so a few very long orders do not squeeze the rest
    widest = float(test_rows['target_h'].quantile(0.98)) * 1.1
    for column, values, colour, title in [
            (column_1, test_rows['mes_remaining_h'], PLAN_GREY, 'MES plan'),
            (column_2, test_rows['ml_pred_h'], MODEL_GREEN, 'Model (p50)')]:
        figure = go.Figure()
        longest = float(max(test_rows['target_h'].max(), values.max())) * 1.05
        figure.add_trace(go.Scatter(x=[0, longest], y=[0, longest], mode='lines',
                                    line=dict(color='#232523', dash='dash', width=1),
                                    name='perfect', hoverinfo='skip'))
        figure.add_trace(go.Scatter(x=test_rows['target_h'], y=values, mode='markers',
                                    marker=dict(color=colour, size=6, opacity=0.55),
                                    name=title, text=test_rows['wo_code'],
                                    hovertemplate='%{text}<br>real %{x:.2f} h<br>said %{y:.2f} h'
                                                  '<extra></extra>'))
        figure.update_layout(
            title=title, height=400, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white',
            showlegend=False,
            xaxis=dict(title='hours really left', range=[0, widest], gridcolor='#E4E7EB'),
            yaxis=dict(title='hours predicted', range=[0, widest], gridcolor='#E4E7EB'))
        column.plotly_chart(figure)
    st.caption('Each dot is one test prediction. On the dashed line the prediction is perfect. '
               'The plan\'s dots sit mostly below the line: it expects too little time. The '
               'model\'s dots sit closer to the line. The view skips the few longest orders; '
               'double-click a chart to see them all.')

    per_checkpoint = []
    for checkpoint in CHECKPOINTS:
        rows_here = test_rows[test_rows['checkpoint'] == checkpoint]
        if len(rows_here) == 0:
            continue
        denominator = rows_here['target_h'].clip(lower=0.5)
        inside = ((rows_here['target_h'] >= rows_here['ml_p10_h']) &
                  (rows_here['target_h'] <= rows_here['ml_p90_h']))
        per_checkpoint.append({
            'checkpoint': '%d%%' % round(checkpoint * 100),
            'plan': float(((rows_here['mes_remaining_h'] - rows_here['target_h']).abs()
                           / denominator * 100).mean()),
            'model': float(((rows_here['ml_pred_h'] - rows_here['target_h']).abs()
                            / denominator * 100).mean()),
            'inside': float(inside.mean() * 100),
        })
    per_checkpoint = pd.DataFrame(per_checkpoint)

    column_1, column_2 = st.columns(2)
    figure = go.Figure()
    figure.add_trace(go.Bar(x=per_checkpoint['checkpoint'], y=per_checkpoint['plan'],
                            marker=dict(color=PLAN_GREY), name='MES plan'))
    figure.add_trace(go.Bar(x=per_checkpoint['checkpoint'], y=per_checkpoint['model'],
                            marker=dict(color=MODEL_GREEN), name='model'))
    figure.update_layout(
        title='Average error at each checkpoint', height=380, barmode='group',
        margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white',
        xaxis=dict(title='share of the ordered pieces already made', type='category'),
        yaxis=dict(title='error, % of the hours really left', gridcolor='#E4E7EB'),
        legend=dict(orientation='h', yanchor='top', y=-0.22, xanchor='left', x=0))
    column_1.plotly_chart(figure)

    figure = go.Figure()
    figure.add_trace(go.Bar(x=per_checkpoint['checkpoint'], y=per_checkpoint['inside'],
                            marker=dict(color=BAND, line=dict(color=MODEL_GREEN, width=1.5)),
                            name='real value inside p10 – p90'))
    figure.add_hline(y=80, line_color='#232523', line_dash='dash')
    figure.add_annotation(x=0, xref='paper', y=80, text='target 80%', showarrow=False,
                          yshift=10, xanchor='left')
    figure.update_layout(
        title='How often the real value falls inside the range', height=380,
        margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white', showlegend=False,
        xaxis=dict(title='share of the ordered pieces already made', type='category'),
        yaxis=dict(title='% of test predictions', range=[0, 100], gridcolor='#E4E7EB'))
    column_2.plotly_chart(figure)
    st.markdown(
        'Overall the model cuts the average error from **%.1f%%** to **%.1f%%** and is closer than '
        'the plan on **%.1f%%** of the test predictions. The range holds the real value **%.1f%%** '
        'of the time, a little below the 80%% it aims at, so it is slightly too narrow.'
        % (result['mes_error_pct'], result['ml_error_pct'], result['win_rate'],
           result['range_coverage']))


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
