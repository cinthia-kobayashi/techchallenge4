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
dados_treinados = joblib.load('dados/dados_treinados_fase4.joblib')

df_bolsa = dados_treinados['df_bolsa_original']
df_bolsa['Data'] = pd.to_datetime(df_bolsa['Data'],format='%d.%m.%Y')
df_bolsa['Var%'] = df_bolsa['Var%'].str.replace("%","")
df_bolsa['Var%'] = df_bolsa['Var%'].str.replace(",",".")
df_bolsa['Var%'] = pd.to_numeric(df_bolsa['Var%'])

sf = dados_treinados['sf']
sf_df = dados_treinados['sf_df']
last_date = dados_treinados['last_date']
last_value = dados_treinados['last_value']
crossvalidation = dados_treinados['crossvalidation']

data_escolhida = datetime.now()
def calc_forecast(data_escolhida):
    deltadata = dados_treinados["last_date"] - data_escolhida
    deltadata = deltadata.days * -1
    deltadata = deltadata + 1
    forecast = dados_treinados["sf"].forecast(df=dados_treinados["sf_df"], h=deltadata)

    forecast['previsao'] = np.where(
    forecast['AutoARIMA'] > forecast['AutoARIMA'].shift(1),
        'subir',
        'descer'
    )

    tamanho_forecast = forecast.shape[0]*-1
    

    acc = accuracy_score(
        forecast['previsao'].dropna(),
        dados_treinados["crossvalidation"][f'actual_trend'][tamanho_forecast:].dropna()
    )
    
    return acc



crossvalidation = dados_treinados['crossvalidation'].copy()
crossvalidation['actual_trend'] = np.where(
    crossvalidation['y'] > crossvalidation['y'].shift(1),
    'subir',
    'descer'
)
crossvalidation['predicted_trend_AutoARIMA'] = np.where(
    crossvalidation['AutoARIMA'] > crossvalidation['y'].shift(1),
    'subir',
    'descer'
)
model_accuracy = accuracy_score(
    crossvalidation['actual_trend'].dropna(),
    crossvalidation['predicted_trend_AutoARIMA'].dropna()
)

#FERIADOS PARA SEREM RETIRADOS DE CONSIDERAÇÃO
feriados_brasil = [
    # 2025

    '2025-12-31',  # Facultativo
    
    # 2026
    '2026-01-01',  # Ano Novo
    '2026-02-16',  # Carnaval
    '2026-02-17',  # Carnaval
    '2026-04-03',  # Sexta-feira Santa
    '2026-04-21',  # Tiradentes
    '2026-05-01',  # Dia do Trabalho
    '2026-06-04',  # Corpus Christi
    '2026-09-07',  # Independência
    '2026-10-12',  # Nossa Senhora Aparecida
    '2026-11-02',  # Finados
    '2026-11-15',  # Proclamação da República
    '2026-12-25',  # Natal
]



# VERIFICAR SE É DIA ÚTIL
def is_dia_util(data, feriados_list):
    """
    Verifica se uma data é dia útil (segunda a sexta e não é feriado)
    """
    # Se data for Timestamp, converter para date
    if isinstance(data, pd.Timestamp):
        data = data.date()
    
    # Verifica se é final de semana (0=segunda, 6=domingo)
    if data.weekday() >= 5:  # 5=sábado, 6=domingo
        return False
    
    # Verifica se é feriado (agora comparando dates)
    if data in feriados_list:
        return False
    
    return True

#Calcular próximos dias úteis
def proximos_dias_uteis(data_inicio, n_dias, feriados_list):
    """
    Retorna as próximas n datas úteis a partir de data_inicio
    """
    datas = []
    data_atual = data_inicio
    
    while len(datas) < n_dias:
        data_atual += timedelta(days=1)
        if is_dia_util(data_atual, feriados_list):
            datas.append(data_atual)
    
    return datas

#Calcular dias úteis entre duas datas
def contar_dias_uteis_entre(data_inicio, data_fim, feriados_list):
    """
    Conta quantos dias úteis há entre data_inicio (exclusive) e data_fim (inclusive)
    """
    contador = 0
    data_atual = data_inicio
    
    while data_atual < data_fim:
        data_atual += timedelta(days=1)
        if is_dia_util(data_atual, feriados_list):
            contador += 1
    
    return contador
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


#Título no topo

st.title("POSTECH - FIAP (Data Analytics - BB)")
st.divider()
with st.expander("TechChallenge Fase 4 - Data viz and production models"):
    st.subheader(
        """TechChallenge Fase 4 - Data viz and production models
        """)

    st.markdown("**PROBLEMA:**")

    st.markdown('''
                
            Transformar o modelo de séries temporais da IBOVESPA, :green-background[desenvolvido no *techchallenge da fase 2*], 
            em uma aplicação interativa utilizando o Streamlit, 
            permitindo que qualquer pessoa insira os dados e veja a previsão em tempo real.
            
                ''')


    st.divider()

    st.markdown("**Explicação:**")

    st.markdown('''
            A Fase 2 foi realizada em agosto de 2025 utilizando o IBOVESPA para treinamento dos dados.
            Na ocasião, o método de treino para predição de dados que ficou com maior acurácia e o escolhido foi o AutoARIMA.

            Para que os dados pudessem ficar com atualização mais recente, decidimos treinar novamente o modelo com dados da bolsa até o dia 30/12/25.
            A acurácia ficou em 86% e o desenvolvimento do treinamento pode ser visto [clicando aqui](/Treino_de_Dados)   
            ''')

st.divider()


st.title("Previsão de Tendência da Bolsa :material/whatshot:")

col1, col2, col3 = st.columns([2,2,6])
col1.metric("Última Data Disponível no Modelo", df_bolsa['Data'].max().strftime('%d/%m/%Y'))
col2.metric("Último Fechamento", f" {df_bolsa['Último'].iloc[-1]}",delta=f" {df_bolsa['Var%'].iloc[-1]}%")
col3.markdown("**Confiança por Horizonte Temporal**")

horizons = list(range(1, 31))  # 1 a 30 dias
confidences = [model_accuracy * np.exp(-0.05 * (h-1)) for h in horizons]

df_confidence = pd.DataFrame({
    'Dias à frente': horizons,
    'Confiança Estimada': confidences
})


confidence_chart = alt.Chart(df_confidence).mark_line(point=True).encode(
    x='Dias à frente:Q',
    y=alt.Y('Confiança Estimada:Q', axis=alt.Axis(format='%')),
    tooltip=['Dias à frente:Q', alt.Tooltip('Confiança Estimada:Q', format='.1%')]
).properties(
    height=300,
    title="Decaimento estimado da confiança com o horizonte temporal"
).configure_axisY(
    grid=False
)

col3.altair_chart(confidence_chart, use_container_width=True)


#--------------------------
#PREVISÃO PRÓXIMO DIA - SEMPRE APARECE
#--------------------------

# Fazer previsão
forecast = sf.forecast(df=sf_df, h=1)
predicted_value = forecast['AutoARIMA'].iloc[0]
pred_date = pd.to_datetime(forecast['ds'].iloc[0]) + timedelta(days=2)



# Determinar tendência
trend = "subir" if predicted_value > last_value else "descer"

percent_change = ((predicted_value - last_value) / last_value) * 100

col1, col2, col3 = st.columns(3)

with col2:
    st.metric(
        "Confiança do Modelo",
        f"{model_accuracy*100:.1f}%"
    )
    st.caption("Acurácia histórica do modelo para previsões de 1 dia")




st.divider()
#--------------------------
#PREVISÃO PARA DATA ESPECÍFICA
#--------------------------

max_reasonable_days = min(90, int(30 / (1 - model_accuracy))) 

st.caption("*Nota: Acurácia para horizontes maiores pode variar.*")



feriados = [pd.to_datetime(data).date() for data in feriados_brasil]


st.subheader("2. 📅 Previsão para Data Específica")

st.caption(f"Próximos feriados: {', '.join([f.strftime('%d/%m') for f in feriados if f >= last_date.date() and f <= last_date.date() + timedelta(days=60)])}")

dias_uteis_disponiveis = proximos_dias_uteis(last_date.date(), max_reasonable_days, feriados)


min_date = dias_uteis_disponiveis[0] if dias_uteis_disponiveis else last_date.date() + timedelta(days=1)
max_date = dias_uteis_disponiveis[-1] if dias_uteis_disponiveis else last_date.date() + timedelta(days=max_reasonable_days)



selected_date = st.date_input(
    "Selecione uma data futura (apenas dias úteis)",
    min_value=min_date,
    max_value=max_date,
    value=min_date,
    format="DD/MM/YYYY"
)


dias_semana_pt = {
    'Monday': 'Segunda-feira',
    'Tuesday': 'Terça-feira', 
    'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira',
    'Friday': 'Sexta-feira',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}


if selected_date:
    dia_semana_en = selected_date.strftime('%A')
    dia_semana_pt = dias_semana_pt.get(dia_semana_en, dia_semana_en)
    
    # Verificar se é feriado específico
    is_feriado = selected_date in feriados
    
    if not is_dia_util(selected_date, feriados):
        if is_feriado:
            st.error(f"❌ {selected_date.strftime('%d/%m/%Y')} ({dia_semana_pt}) é FERIADO!")
            st.info("A bolsa não funciona em feriados. Por favor, selecione um dia útil.")
        elif selected_date.weekday() >= 5:
            st.error(f"❌ {selected_date.strftime('%d/%m/%Y')} ({dia_semana_pt}) é FIM DE SEMANA!")
            st.info("A bolsa não funciona nos finais de semana. Por favor, selecione um dia de semana.")
    else:
        st.success(f"✅ {selected_date.strftime('%d/%m/%Y')} ({dia_semana_pt}) é um dia útil válido!")
        if st.button(f"Prever para {selected_date.strftime('%d/%m/%Y')}", key="predict_specific"):
            with st.spinner("Calculando..."):
                # Calcular dias úteis à frente
                days_ahead_uteis = contar_dias_uteis_entre(last_date.date(), selected_date, feriados)
                
                if days_ahead_uteis <= 0:
                    st.error("Selecione uma data futura!")
                else:
                    # Fazer previsão para DIAS CORRIDOS (seu modelo foi treinado assim)
                    # Precisamos mapear dias úteis para posições no forecast
                    
                    # 1. Calcular quantos dias corridos correspondem a N dias úteis
                    dias_corridos_necessarios = 0
                    dias_uteis_encontrados = 0
                    data_atual = last_date.date()
                    
                    while dias_uteis_encontrados < days_ahead_uteis:
                        dias_corridos_necessarios += 1
                        data_atual += timedelta(days=1)
                        if is_dia_util(data_atual, feriados):
                            dias_uteis_encontrados += 1
                    
                    # 2. Fazer previsão para dias corridos
                    forecast_corridos = sf.forecast(df=sf_df, h=dias_corridos_necessarios)
                    
                    if not forecast_corridos.empty:
                        # 3. Filtrar apenas dias úteis do forecast
                        forecast_dias_uteis = []
                        indice_corrido = 0
                        data_atual = last_date.date()
                        
                        # Garantir que forecast_corridos['ds'] seja datetime
                        forecast_corridos['ds'] = pd.to_datetime(forecast_corridos['ds'])
                        
                        for i in range(len(forecast_corridos)):
                            data_atual += timedelta(days=1)
                            if is_dia_util(data_atual, feriados):
                                forecast_dias_uteis.append({
                                    'ds': data_atual,
                                    'AutoARIMA': forecast_corridos['AutoARIMA'].iloc[i],
                                    'Data_Original': forecast_corridos['ds'].iloc[i]
                                })
                        
                        # Criar DataFrame apenas com dias úteis
                        forecast_df = pd.DataFrame(forecast_dias_uteis)
                        
                        # Verificar se temos previsão para a data selecionada
                        if len(forecast_df) >= days_ahead_uteis:
                            # Pegar previsão para o último dia útil (data selecionada)
                            predicted_value = forecast_df['AutoARIMA'].iloc[-1]
                            last_real_value = last_value
                            
                            # Tendência vs "Hoje"
                            trend_vs_hoje = "📈 Subir" if predicted_value > last_real_value else "📉 Descer"
                            
                            # Tendência vs Dia Anterior (útil)
                            if days_ahead_uteis > 1:
                                predicted_previous = forecast_df['AutoARIMA'].iloc[-2]
                            else:
                                predicted_previous = last_real_value
                            trend_vs_anterior = "📈 Subir" if predicted_value > predicted_previous else "📉 Descer"
                            
                            # Confiança estimada (baseada em dias úteis)
                            estimated_confidence = model_accuracy * np.exp(-0.05 * (days_ahead_uteis - 1))
                            
                            # Mostrar informações
                            st.info(f"""
                            **📊 Informações da previsão:**
                            - **Dias úteis à frente:** {days_ahead_uteis}
                            - **Dias corridos correspondentes:** {dias_corridos_necessarios}
                            - **Data prevista no calendário:** {forecast_df['ds'].iloc[-1].strftime('%d/%m/%Y')}
                            """)
                            
                            # Mostrar resultados principais
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Dias úteis à frente", days_ahead_uteis)
                                st.metric("Data Alvo", selected_date.strftime('%d/%m/%Y'))
                            
                            with col2:
                                st.write("**vs 30/12/2025:**")
                                if "📈" in trend_vs_hoje:
                                    st.success(trend_vs_hoje)
                                else:
                                    st.error(trend_vs_hoje)
                                
                                st.write("**vs Dia Anterior:**")
                                if "📈" in trend_vs_anterior:
                                    st.success(trend_vs_anterior)
                                else:
                                    st.error(trend_vs_anterior)
                            
                            with col3:
                                st.metric(
                                    "Confiança Estimada",
                                    f"{estimated_confidence*100:.1f}%",
                                    delta=f"-{(model_accuracy - estimated_confidence)*100:.1f}%",
                                    help=f"Acurácia base: {model_accuracy*100:.1f}%"
                                )
                                if estimated_confidence < 0.6:
                                    st.warning("⚠️ Baixa confiança")
                            
                            # Tabela detalhada APENAS com dias úteis
                            st.subheader("📊 Análise Dia a Dia (apenas dias úteis)")
                            
                            # Preparar dados para exibição
                            display_data = forecast_df.copy()
                            
                            # Garantir que 'ds' seja datetime
                            display_data['ds'] = pd.to_datetime(display_data['ds'])
                            
                            # Criar colunas formatadas
                            display_data['Data'] = display_data['ds'].dt.strftime('%d/%m/%Y')
                            display_data['Previsão'] = display_data['AutoARIMA']
                            display_data['Dia da Semana'] = display_data['ds'].apply(
                                lambda x: dias_semana_pt.get(x.strftime('%A'), x.strftime('%A'))
                            )
                            
                            # Calcular vs Hoje
                            display_data['vs 30/12/2025'] = display_data['Previsão'].apply(
                                lambda x: "📈 Subir" if x > last_real_value else "📉 Descer"
                            )
                            
                            # Calcular vs Dia Anterior (apenas dias úteis)
                            previsoes = display_data['Previsão'].values
                            valor_anterior = np.insert(previsoes[:-1], 0, last_real_value)
                            comparacoes = previsoes > valor_anterior
                            display_data['vs Dia Anterior'] = np.where(comparacoes, "📈 Subir", "📉 Descer")
                            
                            # Mostrar tabela
                            display_df = display_data[['Data', 'Dia da Semana', 'vs 30/12/2025', 'vs Dia Anterior']]
                            st.dataframe(
                                display_df, 
                                hide_index=True,
                                column_config={
                                    "Data": st.column_config.TextColumn("Data"),
                                    "Dia da Semana": st.column_config.TextColumn("Dia"),
                                    "vs 30/12/2025": st.column_config.TextColumn("Tendência vs 30/12/2025"),
                                    "vs Dia Anterior": st.column_config.TextColumn("Tendência vs Dia Anterior")
                                }
                            )
                            
                            # Gráfico APENAS com dias úteis
                            st.subheader("📈 Visualização das Tendências (apenas dias úteis)")
                            
                            # Adicionar ponto de hoje
                            hoje_data = pd.DataFrame({
                                'ds': [last_date],
                                'Previsão': [last_real_value],
                                'vs Dia Anterior': ['Hoje'],
                                'Dia': ['Hoje'],
                                'Data_Formatada': [last_date.strftime('%d/%m/%Y')]
                            })
                            
                            # Preparar dados para gráfico
                            graph_data = display_data.copy()
                            graph_data['Data_Formatada'] = graph_data['ds'].dt.strftime('%d/%m/%Y')
                            graph_data['Dia'] = graph_data['Dia da Semana']
                            
                            # Combinar
                            combined_data = pd.concat([hoje_data, graph_data], ignore_index=True)
                            
                            # Garantir que 'ds' seja datetime para Altair
                            combined_data['ds'] = pd.to_datetime(combined_data['ds'])
                            
                            # Criar gráfico
                            chart = alt.Chart(combined_data).mark_line(color='blue').encode(
                                x=alt.X('ds:T', title='Data', axis=alt.Axis(format='%d/%m')),
                                y=alt.Y('Previsão:Q', title='Valor Previsto'),
                                tooltip=['Data_Formatada:N', 'Dia:N', 'Previsão:Q']
                            )
                            
                            points = alt.Chart(combined_data[combined_data['vs Dia Anterior'].isin(['📈 Subir', '📉 Descer'])]).mark_point(
                                size=100
                            ).encode(
                                x='ds:T',
                                y='Previsão:Q',
                                color=alt.Color('vs Dia Anterior:N', 
                                                scale=alt.Scale(domain=['📈 Subir', '📉 Descer'],
                                                                range=['green', 'red'])),
                                tooltip=['Data_Formatada:N', 'Dia:N', 'Previsão:Q', 'vs Dia Anterior:N']
                            )
                            
                            hoje_point = alt.Chart(combined_data[combined_data['vs Dia Anterior'] == 'Hoje']).mark_point(
                                size=200, color='black', shape='diamond'
                            ).encode(
                                x='ds:T',
                                y='Previsão:Q',
                                tooltip=['Data_Formatada:N', 'Dia:N', 'Previsão:Q']
                            )
                            
                            final_chart = (chart + points + hoje_point).properties(
                                height=400,
                                title=f"Previsão (apenas dias úteis): {last_date.strftime('%d/%m')} a {selected_date.strftime('%d/%m/%Y')}"
                            )
                            
                            st.altair_chart(final_chart, use_container_width=True)
                            
                            # Mostrar datas excluídas (fins de semana/feriados)
                            if dias_corridos_necessarios > days_ahead_uteis:
                                st.caption(f"*Foram excluídos {dias_corridos_necessarios - days_ahead_uteis} dias não úteis (fins de semana/feriados)*")
                            
                        else:
                            st.error("Não foi possível gerar previsão para a data selecionada.")
                    else:
                        st.error("Erro ao gerar previsão.")

st.divider()

st.markdown("Aqui só uma demonstração de interação com a base de dados real, analisando o IBOVESPA na base que foi treinado o nosso modelo.")

with st.expander("Análise da base real"):

    st.title("IBOVESPA - Dashboard :material/analytics:")

    st.subheader("Histórico da bolsa de 01/06/2023 a 30/12/2025")

    st.write("Selecione o intervalo para analisar no gráfico:")

    col1, col2,col3,col4 = st.columns([3,1,1,3])
    start_date = col2.date_input("Data inicial", value=df_bolsa["Data"].max().replace(day=1),format="DD/MM/YYYY")
    end_date = col3.date_input("Data final", value=df_bolsa["Data"].max(),format="DD/MM/YYYY")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)


    #--------------------------
    #INTERAÇÃO COM A BASE ORIGINAL PARA O USUÁRIO
    #--------------------------
    graf_principal = alt.Chart(df_bolsa).mark_line().encode(
        alt.X("Data:T", title="Data"),
        alt.Y("Último:Q", title="Fechamento", scale=alt.Scale(zero=False))
    ).properties(height=400)

    graf_destaque = alt.Chart(pd.DataFrame({
        'start': [start_date],
        'end': [end_date]
    })).mark_rect(
        opacity=0.3,
        color='lightblue'
    ).encode(
        x='start:T',
        x2='end:T'
    )
    linha_inicial = alt.Chart(pd.DataFrame({'date': [start_date]})).mark_rule(
        color='red',
        strokeWidth=2
    ).encode(x='date:T')

    linha_final = alt.Chart(pd.DataFrame({'date': [end_date]})).mark_rule(
        color='red',
        strokeWidth=2
    ).encode(x='date:T')

    # Combinar todos os gráficos
    graf_final_camadas = alt.layer(
        graf_principal,
        graf_destaque,
        linha_inicial,
        linha_final
    ).properties(height=400)

    # Exibir o gráfico
    col1, col2, col3 = st.columns([70,5,25])
    col1.altair_chart(graf_final_camadas, use_container_width=True)



    df_filtered = df_bolsa[
        (df_bolsa["Data"] >= pd.to_datetime(start_date)) & 
        (df_bolsa["Data"] <= pd.to_datetime(end_date))
    ]
    # Encontrar a data do maior e menor 'Último'
    data_max_ultimo = df_filtered.loc[df_filtered['Último'].idxmax(), 'Data']
    data_min_ultimo = df_filtered.loc[df_filtered['Último'].idxmin(), 'Data']

    # Encontrar a data do maior e menor 'Var%'
    data_max_var = df_filtered.loc[df_filtered['Var%'].idxmax(), 'Data']
    data_min_var = df_filtered.loc[df_filtered['Var%'].idxmin(), 'Data']

    col3.subheader("**Análise do período**")
    col3.markdown("---")

    # Maior Último
    col3.markdown(f"**Maior Valor:** {df_filtered['Último'].max():.2f} | **Data:** {data_max_ultimo.strftime('%d/%m/%Y')}")
    #col2.markdown(f"*Data:* {data_max_ultimo.strftime('%d/%m/%Y')}")

    # Menor Último  
    col3.markdown(f"**Menor Valor:** {df_filtered['Último'].min():.2f} | **Data:** {data_min_ultimo.strftime('%d/%m/%Y')}")
    #col2.markdown(f"*Data:* {data_min_ultimo.strftime('%d/%m/%Y')}")

    # Variação

    col3.markdown(f"**Maior Variação (%):** {df_filtered['Var%'].max():.2f}% | **Data:** {data_max_var.strftime('%d/%m/%Y')}")
    #col2.markdown(f"*Data:* {data_max_var.strftime('%d/%m/%Y')}")

    col3.markdown(f"**Menor Variação (%):** {df_filtered['Var%'].min():.2f}% | **Data:** {data_min_var.strftime('%d/%m/%Y')}")
    #col2.markdown(f"*Data:* {data_min_var.strftime('%d/%m/%Y')}")


    st.subheader("**Dados do período selecionado** :material/table_chart: ")
    st.dataframe(df_filtered,column_config={"Data":st.column_config.DateColumn(format="DD/MM/YYYY")})









