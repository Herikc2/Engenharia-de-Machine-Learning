# Importações
import pandas as pd
import mlflow
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Projeto_Kobe'
model_uri = f"models:/{experiment_name}@staging"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

# Carregar o modelo e os dados de produção
modelo = mlflow.sklearn.load_model(model_uri)
df_prd = pd.read_parquet('../Data/Raw/dataset_kobe_prod.parquet')

# Remoção de linhas com dados faltantes
print("Antes do tratamento de dados faltantes:")
print("\nDimensão dos dados:", df_prd.shape)

print("Número de dados faltantes:")
print(df_prd.isna().sum())

df_prd_filtered = df_prd.dropna()

print("\n\nDepois do tratamento de dados faltantes:")

print("\nDimensão dos dados:", df_prd_filtered.shape)

print("Número de dados faltantes:")
print(df_prd_filtered.isna().sum())

# Separar variaveis do target
data_cols = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
X_prd = df_prd_filtered[data_cols]

# Iniciando Pipepile no MLFLOW
with mlflow.start_run(experiment_id = experiment_id, run_name = 'PipelineAplicacao'):
    # Fazer previsões
    Y_pred_proba = modelo.predict_proba(X_prd)[:, 1]
    
    # Adicionar as previsões aos dados de produção
    df_prd_filtered['predict_score'] = Y_pred_proba

    # Salvar os dados de produção com as previsões
    df_prd_filtered.to_parquet('../Data/Processed/prediction_prd.parquet')
    mlflow.log_artifact('../Data/Processed/prediction_prd.parquet')


###########################################################
#####################    STREAMLIT    #####################
###########################################################

# Carregando DEV e PRD
df_prd = pd.read_parquet('../Data/Processed/prediction_prd.parquet')
df_dev = pd.read_parquet('../Data/Processed/prediction_test.parquet')
target = 'target'

st.title("Monitoramento")
st.markdown("Em homenagem ao jogador da NBA Kobe Bryant (falecido em 2020), foram disponibilizados os dados de 20 anos de arremessos, bem sucedidos ou não, e informações correlacionadas. O objetivo desse estudo é aplicar técnicas de inteligência artificial para prever se um arremesso será convertido em pontos ou não.")
st.markdown("Repositório Github: https://github.com/Herikc2/Engenharia-de-Machine-Learning")

# Matriz de Confusão
st.subheader("Matriz de Confusão")
cm = metrics.confusion_matrix(df_dev.prediction_label, df_dev[target])
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', cbar = False, ax = ax)
ax.set_title('Matriz de Confusão')
ax.set_xlabel('Previsto')
ax.set_ylabel('Real')
st.pyplot(fig)
fignum = plt.figure(figsize = (12, 8))


# Grafico de densidade das bases
st.subheader("Densidade")
sns.distplot(df_prd.predict_score,
             label = 'Producao',
             ax=plt.gca())

sns.distplot(df_dev.prediction_score_1,
             label = 'Teste',
             ax=plt.gca())
plt.title('Monitoramento')
plt.ylabel('Densidade')
plt.xlabel('Probabilidade')
plt.xlim((0, 1))
plt.grid(True)
plt.legend(loc = 'best')
st.pyplot(fignum)

# Relatório de Classificação
st.subheader("Relatório")
report = pd.DataFrame(metrics.classification_report(df_dev.prediction_label, df_dev[target], output_dict = True))
report.rename(columns = {'0': 'Errou', '1': 'Acertou'}, inplace = True)
report = report[['Errou', 'Acertou']]
st.write(report)