import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_confusion_matrix_from_dict(data,
                                    group_names=('Verdadeiro Negativo', 'Falso Positivo', 'Falso Negativo', 'Verdadeiro Positivo'),
                                    categories=('Zero', 'Um'),
                                    count=True,
                                    percent=True,
                                    cbar=True,
                                    figsize=(6.5, 4.5),
                                    cmap='binary',
                                    box_aspect=0.5, # Define a proporção (altura / largura) dos quadrantes
                                    title=None):
    """
    Gera a Matriz de Confusão a partir do dicionário de entrada com as métricas.
    """
    # 1. Extração dos valores da Matriz de Confusão
    cf = np.array([[data['TN'], data['FP']],
                   [data['FN'], data['TP']]])

    # 2. Formatação dos textos dos quadrantes (Rótulo, Contagem com separador de milhar e Porcentagem)
    blanks = ['' for _ in range(cf.size)]
    
    group_labels = [f"{value}\n" for value in group_names] if group_names and len(group_names) == cf.size else blanks
    
    # Formata números grandes com separador de milhar (ex: 84,782,941)
    group_counts = [f"{value:,}\n" for value in cf.flatten()] if count else blanks
    
    # Calcula as porcentagens relativas ao total de pontos
    total = data.get('total_pontos', np.sum(cf))
    group_percentages = [f"{value / total:.2%}" for value in cf.flatten()] if percent else blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # 3. Extração das estatísticas de resumo
    accuracy = data.get('accuracy', np.trace(cf) / float(np.sum(cf)))
    recall = data.get('recall', cf[1, 1] / sum(cf[1, :]))
    aucroc = data.get('aucroc')
    # precision = data.get('precision', cf[1, 1] / sum(cf[:, 1]))
    # f1_score = data.get('f1', 2 * (precision * recall) / (precision + recall))

    stats_text = (f"\n\nAcurácia={accuracy:0.3f}\n"
                  f"Recall={recall:0.3f}\n"
                  f"AUC-ROC Score={aucroc:0.3f}\n")
    #              f"Precisão={precision:0.3f}\n"
    #              f"F1 Score={f1_score:0.3f}")

    # 4. Plotagem com Seaborn
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=False, cbar_kws={'shrink': box_aspect * 1.6} if cbar else None, # Ajusta a barra lateral para acompanhar a altura do gráfico
        xticklabels=categories, yticklabels=categories, linewidths=0.8, linecolor='black', ax=ax)

    # Reduz a altura dos quadrantes em relação à largura (0.5 = altura é metade da largura)
    ax.set_box_aspect(box_aspect)

    # 5. Adiciona a barra de cores alinhada EXATAMENTE com a altura da matriz
    if cbar:
        cax = inset_axes(
            ax, 
            width="3.5%",          # Largura da barra de cores
            height="100%",         # 100% da altura da matriz
            loc='lower left',
            bbox_to_anchor=(1.03, 0., 1, 1), # Posiciona a 3% de distância da borda direita da matriz
            bbox_transform=ax.transAxes, 
            borderpad=0
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cf.min(), vmax=cf.max()))
        sm.set_array([])
        fig.colorbar(sm, cax=cax)

    # Eixos e Títulos
    ax.set_ylabel('Rótulo Real')
    ax.set_xlabel('Rótulo Predito' + stats_text)

    # Define o título (usa a 'tag' do dicionário se nenhum título for passado)
    plot_title = title if title else f"Matriz de Confusão: {data.get('tag', '')}"
    ax.set_title(plot_title)

    # plt.savefig('matriz_confusao_alinhada.png', bbox_inches='tight', dpi=150)
    plt.show()


# --- Exemplo de Uso ---

# Seus dados de entrada (pode carregar de um arquivo .json com json.load)
data_input = {'cen1': {'titulo': 'Matriz de Confusão - Cenário I\nCentralizado / Real',
          'tag': 'pi1_esp_v0real',
          'TP': 546540,
          'TN': 74000230,
          'FP': 528114,
          'FN': 3734516,
          'total_pontos': 78809400,
          'accuracy': 0.945912,
          'recall': 0.127665,
          'aucroc': 0.852545,
          'precision': 0.508573,
          'f1': 0.204096,
          'pr_auc': 0.303471,
          'taxa_anomalia_teste': 0.054322,
          'accuracy_baseline_majoritaria': 0.945678,
          'threshold': 0.5,
          'pis': [1, 2, 3, 4, 5],
          'max_shards': 15,
          'max_windows': 0,
          'ckpt_sha256': 'ff393b0397b2972ea9ad35ef4099ce40303ae4c9dba9796a37f38d15cb4fdbd0',
          'init_checkpoint': '/mnt/juliana-truenas/Synth-EnergyBench-Anomaly/04_models/v0_real/best_model.pth',
          'wall_time_s': 15297.3,
          'wall_time_treino_s': 14466.1,
          'epocas_executadas': 7,
          'val_mode': 'atual',
          'cpu_pct_avg': 91.72671252989413,
          'load1_avg': 3.691679917041134,
          'ram_used_gb_avg': 1.4215295292537518,
          'ram_used_gb_max': 2.2395782470703125},
 'cen2': {'titulo': 'Matriz de Confusão - Cenário II\nFederado / Real',
          'tag': 'full_real',
          'TP': 3687292,
          'TN': 50111374,
          'FP': 24416970,
          'FN': 593764,
          'total_pontos': 78809400,
          'accuracy': 0.682643,
          'recall': 0.861304,
          'aucroc': 0.853151,
          'precision': 0.1312,
          'f1': 0.227714,
          'pr_auc': 0.297813,
          'taxa_anomalia_teste': 0.054322,
          'accuracy_baseline_majoritaria': 0.945678,
          'threshold': 0.5,
          'pis': [1, 2, 3, 4, 5],
          'max_shards': 15,
          'max_windows': 0,
          'ckpt_sha256': '20864f5ca8d2bab08325bd83af1cf75563c2f7d9a8de36407be9a61dcc8c60a6'},
 'cen3': {'titulo': 'Matriz de Confusão - Cenário III\nCentralizado / Real + Sintético',
          'tag': 'pi1_esp_v0both',
          'TP': 512625,
          'TN': 74047979,
          'FP': 480365,
          'FN': 3768431,
          'total_pontos': 78809400,
          'accuracy': 0.946088,
          'recall': 0.119743,
          'aucroc': 0.850457,
          'precision': 0.516244,
          'f1': 0.194395,
          'pr_auc': 0.301557,
          'taxa_anomalia_teste': 0.054322,
          'accuracy_baseline_majoritaria': 0.945678,
          'threshold': 0.5,
          'pis': [1, 2, 3, 4, 5],
          'max_shards': 15,
          'max_windows': 0,
          'ckpt_sha256': '7bcf49903e00772220a2419cef108b9fdabda199502fe62aa5a333cbc9441936',
          'init_checkpoint': '/mnt/juliana-truenas/Synth-EnergyBench-Anomaly/04_models/v0_final/best_model.pth',
          'wall_time_s': 21967.0,
          'wall_time_treino_s': 21076.5,
          'epocas_executadas': 10,
          'val_mode': 'atual',
          'cpu_pct_avg': 93.36026418969728,
          'load1_avg': 3.738504983388704,
          'ram_used_gb_avg': 1.4254310691192584,
          'ram_used_gb_max': 2.2830276489257812},
 'cen4': {'titulo': 'Matriz de Confusão - Cenário IV\nFederado / Real + Sintético',
          'tag': 'full_both',
          'TP': 3651758,
          'TN': 50889078,
          'FP': 23639266,
          'FN': 629298,
          'total_pontos': 78809400,
          'accuracy': 0.69206,
          'recall': 0.853004,
          'aucroc': 0.853207,
          'precision': 0.133808,
          'f1': 0.231328,
          'pr_auc': 0.297664,
          'taxa_anomalia_teste': 0.054322,
          'accuracy_baseline_majoritaria': 0.945678,
          'threshold': 0.5,
          'pis': [1, 2, 3, 4, 5],
          'max_shards': 15,
          'max_windows': 0,
          'ckpt_sha256': '11ab03d86c26a5d35f624bb1c36270043d770255146cf9c4f08d6185c34e1e7b'}
}

# Gerar o gráfico
cenario="cen1"
plot_confusion_matrix_from_dict(data_input[cenario], title=data_input[cenario]["titulo"], cmap='crest', box_aspect=0.4)

# escalas de cor (cmap): binary, Reds, Blues, crest, viridis, coolwarm, magma, cividis, inferno, plasma, cubehelix, rocket, icefire
