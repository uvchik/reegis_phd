import logging
import reegis_tools.config as cfg


ORDER_KEYS = ['hydro', 'geothermal', 'solar', 'pv', 'wind', 'chp', 'hp', 'pp',
            'import', 'shortage', 'power_line', 'demand',
            'heat_elec_decentralised', 'storage', 'export', 'excess']


def get_orderlist_from_multiindex(index, orderkeys=None):
    """Create an order list by searching the label for key words"""
    order = []

    if orderkeys is None:
        orderkeys = ORDER_KEYS

    index = list(index)

    for element in orderkeys:
        tmp = [x for x in index if element in str(x).lower()]
        for t in tmp:
            index.remove(t)
        order.extend(tmp)
    return order


def get_orderlist(my_node, flow=None):
    """Create an order list by searching the label for key words

    my_node : pd.DataFrame
        Sequence DataFrame of a node view.

    flow : str
        Direction of the flow 'in' or 'out'.

    """
    cols = list(my_node.columns)
    if flow == 'in':
        f = 0
    elif flow == 'out':
        f = 1
    else:
        logging.error("A flow has to be 'in' or 'out.")
        f = None
    order = []

    for element in ORDER_KEYS:
        tmp = [x for x in cols if element in x[0][f].lower()]
        for t in tmp:
            cols.remove(t)
        order.extend(tmp)
    return order


def get_cdict(my_node):
    my_colors = cfg.get_dict_list('plot_colors', string=True)
    color_dict = {}
    for col in my_node.columns:
        n = 0
        color_keys = list(my_colors.keys())

        try:
            while color_keys[n] not in str(col).lower():
                n += 1
            if len(my_colors[color_keys[n]]) > 1:
                color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
            else:
                color = '#{0}'.format(my_colors[color_keys[n]][0])
            color_dict[col] = color
        except IndexError:
            n = 0
            try:
                while color_keys[n] not in str(col).lower():
                    n += 1
                if len(my_colors[color_keys[n]]) > 1:
                    color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
                else:
                    color = '#{0}'.format(my_colors[color_keys[n]][0])
                color_dict[col] = color
            except IndexError:
                color_dict[col] = '#ff00f0'

    return color_dict


def get_cdict_df(df):
    my_colors = cfg.get_dict_list('plot_colors', string=True)
    color_dict = {}
    for col in df.columns:
        n = 0
        color_keys = list(my_colors.keys())
        try:
            while color_keys[n] not in str(col).lower():
                n += 1
            if len(my_colors[color_keys[n]]) > 1:
                color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
            else:
                color = '#{0}'.format(my_colors[color_keys[n]][0])
            color_dict[col] = color
        except IndexError:
            color_dict[col] = '#ff00f0'
    return color_dict
