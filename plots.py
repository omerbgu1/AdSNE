import matplotlib.pyplot as plt
from constants import DEFAULT_PLT_COLORS


def plot_lp_loss(lp_epochs, lp_baseline_loss_arr, lp_subgraph_aware_loss_arr, dataset_name, k_hop):
    epochs_range = range(1, lp_epochs + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs_range, lp_baseline_loss_arr, label='baseline')
    ax.plot(epochs_range, lp_subgraph_aware_loss_arr, label='subgraph aware')
    ax.set_yscale('log')
    ax.set_title(f'link prediction loss: {dataset_name}' + f', k-hop: {k_hop}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('binary cross entropy loss (log-scale)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.show()


def plot_lp_auc(node_accuracy, lp_baseline_auc, lp_subgraph_aware_auc_list, dataset_name, k_hops_list):
    title_pad = 13
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    competition_node_acc_dict = {
        'Cora': 0.815,
        'CiteSeer': 0.703,
        'PubMed': 0.790,
    }
    try:
        competition_node_acc = competition_node_acc_dict[dataset_name]
    except KeyError:
        bars = ax.bar(['GCN'], [node_accuracy],
                      color=DEFAULT_PLT_COLORS[:1])
    else:
        bars = ax.bar(['GCN', 'GCN\n(competition,\nKipf 2017)'], [node_accuracy, competition_node_acc],
                      color=DEFAULT_PLT_COLORS[:2])
    for bar in bars:
        y_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y_val + 0.02, f'{y_val:.4f}', ha='center', va='bottom')
    ax.set_title('node prediction accuracy', pad=title_pad)
    # ax.tick_params(axis='x', labelrotation=labelrotation)

    ax = axs[1]
    competition_lp_dict = {
        'Cora': (0.9822, 'NESS 2023'),
        'CiteSeer': (0.9943, 'NESS 2023'),
        'PubMed': (0.9667, 'NESS 2023'),
    }
    try:
        competition_lp = competition_lp_dict[dataset_name]
    except KeyError:
        competition_lp = None
    names = ['baseline'] + [f'subgraph\naware\nk={k_hop}' for k_hop in k_hops_list]
    bars_vals = [lp_baseline_auc] + lp_subgraph_aware_auc_list
    if competition_lp:
        names.append(f'competition\n({competition_lp[1]})')
        bars_vals.append(competition_lp[0])
    for i in range(0, len(names), 2):
        print(names[i])
        names[i] = '\n\n' + names[i]
        print(names[i])
    num_bars = len(names)
    colors = DEFAULT_PLT_COLORS[2:num_bars + 2]
    bars = ax.bar(names, bars_vals, color=colors)
    for bar in bars:
        y_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y_val + 0.02, f'{y_val:.4f}', ha='center', va='bottom')
    # ax.set_xlabel('model')
    ax.set_title('link prediction AUC', pad=title_pad)
    # for ax in axs:
    #     ax.set_ylim(0.0, 1.0)
    #     ax.grid(axis='y')
    # ax.tick_params(axis='x', labelrotation=labelrotation)

    dataset_name_and_cite_dict = {
        'Cora': 'Cora [McCallum et al. 2000]',
        'CiteSeer': 'CiteSeer [Giles et al. 1998]',
        'PubMed': 'PubMed [Namata et al. 2012]',
    }
    try:
        suptitle = dataset_name_and_cite_dict[dataset_name]
    except KeyError:
        suptitle = dataset_name
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.show()
