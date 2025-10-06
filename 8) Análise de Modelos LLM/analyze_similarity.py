
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Configurações do modelo
MODELS = {
    "BERTimbau": "neuralmind/bert-base-portuguese-cased",
    "S-BERT": "paraphrase-multilingual-mpnet-base-v2",
    "DistilBERT": "distiluse-base-multilingual-cased-v2"
}

# Carregar os arquivos
FILE_A = "PM-2-Objetivos Estratégicos.xlsx"
FILE_B = "PE-1-ODS.xlsx"

def load_data(file_a, file_b):
    df_a = pd.read_excel(file_a, sheet_name="Planilha1")
    df_b = pd.read_excel(file_b, sheet_name="Planilha1")
    return df_a.iloc[:, 0].dropna().tolist(), df_b.iloc[:, 0].dropna().tolist()

def compute_similarities(sentences_a, sentences_b, model_name):
    print(f"Processando com o modelo: {model_name}")
    model = SentenceTransformer(MODELS[model_name])
    embeddings_a = model.encode(sentences_a, convert_to_tensor=True)
    embeddings_b = model.encode(sentences_b, convert_to_tensor=True)

    results_a_to_b = []
    for i, emb_a in enumerate(embeddings_a):
        similarities = util.pytorch_cos_sim(emb_a, embeddings_b).flatten()
        sorted_indices = similarities.argsort(descending=True)
        results_a_to_b.append([
            (j + 1, sentences_b[j], similarities[j].item()) for j in sorted_indices
        ])

    results_b_to_a = []
    for i, emb_b in enumerate(embeddings_b):
        similarities = util.pytorch_cos_sim(emb_b, embeddings_a).flatten()
        sorted_indices = similarities.argsort(descending=True)
        results_b_to_a.append([
            (j + 1, sentences_a[j], similarities[j].item()) for j in sorted_indices
        ])

    return results_a_to_b, results_b_to_a

def save_results(results_a_to_b, results_b_to_a, model_name, sentences_a, sentences_b):
    # Salvar resultados A → B
    data_a_to_b = []
    for i, row in enumerate(results_a_to_b):
        a_row = [sentences_a[i]] + [f"{idx}: {sent} ({score:.2f})" for idx, sent, score in row]
        data_a_to_b.append(a_row)
    df_a_to_b = pd.DataFrame(data_a_to_b)
    df_a_to_b.to_excel(f"Resultados_A_to_B_{model_name}.xlsx", index=False, header=["Frase A"] + [f"Similaridade {j+1}" for j in range(len(data_a_to_b[0]) - 1)])

    # Salvar resultados B → A
    data_b_to_a = []
    for i, row in enumerate(results_b_to_a):
        b_row = [sentences_b[i]] + [f"{idx}: {sent} ({score:.2f})" for idx, sent, score in row]
        data_b_to_a.append(b_row)
    df_b_to_a = pd.DataFrame(data_b_to_a)
    df_b_to_a.to_excel(f"Resultados_B_to_A_{model_name}.xlsx", index=False, header=["Frase B"] + [f"Similaridade {j+1}" for j in range(len(data_b_to_a[0]) - 1)])

def main():
    sentences_a, sentences_b = load_data(FILE_A, FILE_B)
    for model_name in MODELS:  # Itera sobre BERTimbau, S-BERT e DistilBERT
        results_a_to_b, results_b_to_a = compute_similarities(sentences_a, sentences_b, model_name)
        save_results(results_a_to_b, results_b_to_a, model_name, sentences_a, sentences_b)
        print(f"Resultados salvos para o modelo {model_name}.")

if __name__ == "__main__":
    main()
