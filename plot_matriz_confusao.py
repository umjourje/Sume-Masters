import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_from_dict(data,
                                    group_names=('Verdadeiro Negativo', 'Falso Positivo', 'Falso Negativo', 'Verdadeiro Positivo'),
                                    categories=('Zero', 'Um'),
                                    count=True,
                                    percent=True,
                                    cbar=True,
                                    figsize=(7, 6),
                                    cmap='binary',
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
    precision = data.get('precision', cf[1, 1] / sum(cf[:, 1]))
    recall = data.get('recall', cf[1, 1] / sum(cf[1, :]))
    f1_score = data.get('f1', 2 * (precision * recall) / (precision + recall))

    stats_text = (f"\n\nAcurácia={accuracy:0.3f}\n"
                  f"Precisão={precision:0.3f}\n"
                  f"Recall={recall:0.3f}\n"
                  f"F1 Score={f1_score:0.3f}")

    # 4. Plotagem com Seaborn
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories, linewidths=0.8, linecolor='black')

    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Predito' + stats_text)

    # Define o título (usa a 'tag' do dicionário se nenhum título for passado)
    plot_title = title if title else f"Matriz de Confusão: {data.get('tag', '')}"
    plt.title(plot_title)

    plt.tight_layout()
    plt.show()


# --- Exemplo de Uso ---

# Seus dados de entrada (pode carregar de um arquivo .json com json.load)
data_input = {
  "cen1": {
    "titulo": "Matriz de Confusão - Cenário I\nCentralizado / Real",
    "tag": "pi1_esp_v0real",
    "TP": 383629,
    "TN": 84782941,
    "FP": 820472,
    "FN": 1302230,
    "total_pontos": 87289272,
    "precision": 0.318602,
    "recall": 0.227557,
    "f1": 0.265491,
    "accuracy": 0.975682,
    "taxa_anomalia_teste": 0.019313,
    "wall_time_s": 15075.4,
    "wall_time_treino_s": 14194.2,
    "wall_time_teste_s": 881.2,
    "epocas_executadas": 7,
    "val_mode": "atual",
    "cpu_pct_avg": 94.48707153626775,
    "load1_avg": 3.7439182522903454,
    "ram_used_gb_avg": 1.4162518499265513,
    "ram_used_gb_max": 2.4088592529296875,
  },
  "cen2":{},
  "cen3":{
    "titulo": "Matriz de Confusão - Cenário III\nCentralizado / Real + Sintético",
    "tag": "pi1_esp_v0both",
    "TP": 366338,
    "TN": 84838212,
    "FP": 765201,
    "FN": 1319521,
    "total_pontos": 87289272,
    "precision": 0.323752,
    "recall": 0.2173,
    "f1": 0.260054,
    "accuracy": 0.976117,
    "taxa_anomalia_teste": 0.019313,
    "wall_time_s": 21522.3,
    "wall_time_treino_s": 20649.9,
    "wall_time_teste_s": 872.5,
    "epocas_executadas": 10,
    "val_mode": "atual",
    "cpu_pct_avg": 94.45225215420248,
    "load1_avg": 3.7599806248486316,
    "ram_used_gb_avg": 1.3693734850598236,
    "ram_used_gb_max": 2.259998321533203,},
  "cen4":{},
}

# Gerar o gráfico
cenario="cen1"
plot_confusion_matrix_from_dict(data_input[cenario], title=data_input[cenario]["titulo"], cmap='crest')

# escalas de cor (cmap): binary, Reds, Blues, crest, viridis, coolwarm, magma, cividis, inferno, plasma, cubehelix, rocket, icefire