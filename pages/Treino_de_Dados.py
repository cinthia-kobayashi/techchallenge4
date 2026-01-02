# Bibliotecas
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime, timedelta
import altair as alt

#-------------------------
#CARREGAMENTO DOS DADOS E FUNÇÕES
#--------------------------
dados_treinados = joblib.load('../dados/dados_treinados_fase4.joblib')
df_bolsa = dados_treinados['df_bolsa_original']
df_bolsa['Data'] = pd.to_datetime(df_bolsa['Data'], format='%d.%m.%Y')
df_bolsa['Var%'] = df_bolsa['Var%'].str.replace("%","")
df_bolsa['Var%'] = df_bolsa['Var%'].str.replace(",",".")
df_bolsa['Var%'] = pd.to_numeric(df_bolsa['Var%'])
resultados = dados_treinados['resultados_seasonal_decompose']
result = dados_treinados['result_adfuller']
df_diff = dados_treinados['df_log']
result_diff = dados_treinados['result_diff']
ma_diff = df_diff.rolling(12).mean()
std_diff = df_diff.rolling(12).std()

#-------------------------
#CONFIG DA PÁGINA
#--------------------------
st.set_page_config(
    page_title="Tech Challenge - FIAP",
    page_icon=":material/database:",
    layout="wide"

)

#Colocar o nome do grupo no Barra Lateral.
with st.sidebar:
    st.markdown(''' Grupo:  
        Agnes Miki Magario  
        Cinthia Mayumi Kobayashi  
        Lina Satie Kobata Felippe
 ''')

st.title("POSTECH - FIAP (Data Analytics - BB)")
st.divider()

st.header(
    """TechChallenge Fase 4 - Data viz and production models
    """)

st.subheader("**Demonstração do treino de dados:**")

st.markdown('''
            
        Os dados foram retirados do site [Investing.com](https://br.investing.com/indices/bovespa-historical-data), utilizando a data de **01/06/2023 a 30/12/2025.**    
        Foi utilizado para esse estudo o intervalo de tempo diário e a base de dados dos últimos 2 anos e meio, aproximadamente.   
        O banco de dados histórico do IBOVESPA é extenso, contudo, foi escolhido analisar a tendência de mercado dos últimos anos no pós pandemia, visto que o mercado de ações 
        durante a COVID-19 se comportou de forma atípica.  
        A pandemia foi oficialmente encerrada pela OMS em maio/2023.
            

        O banco de dados possui 7 colunas, abaixo explicadas: 
            

                Data: O dia analisado. 
                Último: A cotação de fechamento do dia. 
                Abertura: A cotação de abertura do dia. 
                Máxima: A cotação máxima atingida do dia. 
                Mínima: A cotação mínima atingida do dia. 
                Vol.: Volume de ações negociadas no IBOVESPA 
                Var%: O percentual de variação entre a cotação de fechamento do dia anterior com a cotação de abertura do dia analisado. 
            ''')

st.markdown("**Dataframe utilizado: :material/table:**")

st.dataframe(df_bolsa, column_config={"Data":st.column_config.DateColumn(format="DD/MM/YYYY")})

st.header("**TREINAMENTO DA BASE**")

st.markdown("A base de dados foi analisada e treinada com as seguintes técnicas demonstradas abaixo")



st.subheader(''' Gráfico da Base original da IBOVESPA''')
st.markdown("Visualmente percebe-se que não se trata de uma série estacionária, mas mesmo assim aplicou-se o teste ADFULLER para certificação")

graf_principal = alt.Chart(df_bolsa).mark_line().encode(
    alt.X("Data:T", title="Data"),
    alt.Y("Último:Q", title="Fechamento", scale=alt.Scale(zero=False))
).properties(height=400)

col1, col2, col3 = st.columns([5,1,5])
with col1:
    st.markdown("Gráfico")
    st.altair_chart(graf_principal)
with col3:
    st.subheader(" Teste de Estacionariedade (ADF) - Série Original")

    # Métricas em colunas
    #metric1, metric2, metric3 = 

    #with metric1:
    st.metric(
        "Teste Estatístico",
        f"{result[0]:.4f}"
    )

    #with metric2:
    st.metric(
        "P-Value",
        f"{result[1]:.4f}"
    )

    #with metric3:
    # Verificar se rejeita H0 (série é estacionária)
    is_stationary = result[1] < 0.05
    status = "✅ Estacionária" if is_stationary else "⚠️ Não Estacionária"
    st.metric("Status", status)

    # Tabela de valores críticos
    st.markdown("#### Valores Críticos:")
    crit_values = pd.DataFrame([
        {'Nível de Significância': '1%', 'Valor Crítico': result[4]['1%']},
        {'Nível de Significância': '5%', 'Valor Crítico': result[4]['5%']},
        {'Nível de Significância': '10%', 'Valor Crítico': result[4]['10%']}
    ])

    st.dataframe(crit_values, hide_index=True)

    # Comparação com valores críticos
    st.markdown("#### 📈 Comparação com Valores Críticos:")

    comparison_data = pd.DataFrame({
        'Nível': ['1%', '5%', '10%'],
        'Valor Crítico': [result[4]['1%'], result[4]['5%'], result[4]['10%']],
        'Teste Estatístico': [result[0], result[0], result[0]],
        'Resultado': [
            "Rejeita H0" if result[0] < result[4]['1%'] else "Não Rejeita",
            "Rejeita H0" if result[0] < result[4]['5%'] else "Não Rejeita", 
            "Rejeita H0" if result[0] < result[4]['10%'] else "Não Rejeita"
        ]
    })




st.markdown('')
col1, col2, col3 = st.columns([5,1,5])
with col1:

    st.markdown("Decomposição Sazonal")
    st.markdown("Analisando a sazonalidade da bolsa, percebendo-se que há uma tendência sazonalidade forte, ajudando o modelo de treino")

    # Converter para DataFrame
    decompose_data = pd.DataFrame({
        'Data': resultados.observed.index,
        'Original': resultados.observed.values,
        'Tendência': resultados.trend.values,
        'Sazonalidade': resultados.seasonal.values,
        'Resíduos': resultados.resid.values
    })

    # Criar gráficos sem scale='independent' e com zero=False
    chart_original = alt.Chart(decompose_data).mark_line().encode(
        x='Data:T',
        y=alt.Y('Original:Q', title='Valor', scale=alt.Scale(zero=False)),
        color=alt.value('blue'),
        tooltip=['Data:T', 'Original:Q']
    ).properties(height=75, title='Série Original')

    chart_trend = alt.Chart(decompose_data).mark_line(color='orange').encode(
        x='Data:T',
        y=alt.Y('Tendência:Q', title='Valor', scale=alt.Scale(zero=False)),
        tooltip=['Data:T', 'Tendência:Q']
    ).properties(height=75, title='Tendência')

    chart_seasonal = alt.Chart(decompose_data).mark_line(color='green').encode(
        x='Data:T',
        y=alt.Y('Sazonalidade:Q', title='Valor', scale=alt.Scale(zero=False, padding=0.5)),
        tooltip=['Data:T', 'Sazonalidade:Q']
    ).properties(height=75, title='Sazonalidade')

    chart_resid = alt.Chart(decompose_data).mark_line(color='red').encode(
        x='Data:T',
        y=alt.Y('Resíduos:Q', title='Valor', scale=alt.Scale(zero=False, padding=0.5)),
        tooltip=['Data:T', 'Resíduos:Q']
    ).properties(height=75, title='Resíduos')

    # Combinar
    final_chart = alt.vconcat(chart_original, chart_trend, chart_seasonal, chart_resid)
    st.altair_chart(final_chart, use_container_width=True)

   

with col3:
    st.markdown("Boxplot - Distribuição do Valor de Fechamento")
    st.markdown("Boxplot sem outliers, reforçando também que não há desvios fortes no comportamento da bolsa.")

    boxplot = alt.Chart(df_bolsa).mark_boxplot(extent='min-max', size=50).encode(
        y=alt.Y('Último:Q', 
                title='Valor de Fechamento (R$)', 
                scale=alt.Scale(zero=False)),
        color=alt.value('steelblue'),
        tooltip=[
            alt.Tooltip('Último:Q', title='Fechamento'),
            alt.Tooltip('Data:T', title='Período', format='%d/%m/%Y')
        ]
    ).properties(
        height=400,
        title='Distribuição dos Valores de Fechamento do IBOVESPA'
    )

    st.altair_chart(boxplot, use_container_width=True)

st.divider()

st.subheader("Normalizando a série:")
st.markdown("Após duas logaritimizações, a série ficou:")
col1, col2, col3 = st.columns([5,1,5])
with col1:

    # Preparar dados para Altair
    analysis_data = pd.DataFrame({
        'Data': df_diff.index,
        'Diferença': df_diff.iloc[:, 0].values,
        'Média_Móvel': ma_diff.iloc[:, 0].values,
        'Desvio_Padrão': std_diff.iloc[:, 0].values,
        'Limite_Superior': ma_diff.iloc[:, 0].values + std_diff.iloc[:, 0].values,
        'Limite_Inferior': ma_diff.iloc[:, 0].values - std_diff.iloc[:, 0].values
    }).dropna()

    # Gráfico principal
    base = alt.Chart(analysis_data).encode(
        x=alt.X('Data:T', title='Data', axis=alt.Axis(format='%b %Y'))
    )

    # Linha das diferenças
    line_diff = base.mark_line(color='blue', opacity=0.7).encode(
        y=alt.Y('Diferença:Q', title='Valor'),
        tooltip=['Data:T', 'Diferença:Q']
    )

    # Linha da média móvel
    line_ma = base.mark_line(color='red', strokeWidth=2).encode(
        y='Média_Móvel:Q',
        tooltip=['Data:T', 'Média_Móvel:Q']
    )

    # Área do desvio padrão (banda)
    area_std = base.mark_area(
        color='green',
        opacity=0.2
    ).encode(
        y='Limite_Superior:Q',
        y2='Limite_Inferior:Q',
        tooltip=['Data:T', 'Limite_Superior:Q', 'Limite_Inferior:Q']
    )

    # Combinar gráficos
    chart = (area_std + line_diff + line_ma).properties(
        height=400,
        title='Diferenças com Média Móvel e Banda de Desvio Padrão'
    ).configure_legend(
        orient='bottom',
        title=None
    )

    st.altair_chart(chart, use_container_width=True)

with col3:
    st.subheader(" Teste de Estacionariedade (ADF)")


    st.metric(
        "Teste Estatístico",
        f"{result_diff[0]:.4f}"
    )

    st.metric(
        "P-Value",
        f"{result_diff[1]:.4f}"
    )

    # Verificar se rejeita H0 (série é estacionária)
    is_stationary = result_diff[1] < 0.05
    status = "✅ Estacionária" if is_stationary else "⚠️ Não Estacionária"
    st.metric("Status", status)

    # Tabela de valores críticos
    st.markdown("#### Valores Críticos:")
    crit_values = pd.DataFrame([
        {'Nível': '1%', 'Valor': result_diff[4]['1%']},
        {'Nível': '5%', 'Valor': result_diff[4]['5%']},
        {'Nível': '10%', 'Valor': result_diff[4]['10%']}
    ])

    st.dataframe(crit_values, hide_index=True)

st.divider()


st.markdown("Com esses dados, treinamos com o modelo autoarima que conta na página principal desse dashboard.")
