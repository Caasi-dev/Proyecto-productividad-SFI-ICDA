import dash
from dash import html, dcc, Input, Output, State
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Carga de datos
path = r"F:\Tareas Isaac\Proyecto DS1\datos_productividad2.csv"
df = pd.read_csv(path)

df_corr = df.copy()
df_corr['Hire_Date'] = pd.to_datetime(df['Hire_Date'])

# Codificar las columnas categóricas utilizando LabelEncoder
columns_to_convert = ['Department', 'Gender', 'Job_Title', 'Education_Level', 'Resigned']
label = LabelEncoder()

# Codificar todas las columnas categóricas
for col in columns_to_convert:
    df_corr[col] = label.fit_transform(df_corr[col])

# Eliminar la segunda codificación de 'Job_Title'
# df_corr['Job_Title_encoded'] = label.fit_transform(df_corr['Job_Title'])  # Esta línea es innecesaria

data = df_corr[['Performance_Score', 'Job_Title', 'Monthly_Salary']]
x = data.drop('Performance_Score', axis=1)
y = data['Performance_Score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de Árbol de Decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Predicción y evaluación del modelo
y_pred = model.predict(x_test)
pres = metrics.accuracy_score(y_test, y_pred)
matriz = metrics.confusion_matrix(y_test, y_pred)


# Gráficas principales
fig_1 = px.histogram(df, x='Work_Hours_Per_Week', nbins=30, title='Histograma de Horas trabajadas a la semana')
fig_1.update_layout(template='plotly_dark')

fig_2 = px.scatter(df, x='Monthly_Salary', y='Performance_Score', color='Performance_Score', title='Rendimiento vs Salario mensual')
fig_2.update_layout(template='plotly_dark')

fig_3 = px.scatter(df, x='Monthly_Salary', y='Performance_Score', color='Job_Title', title='Performance_Score vs Monthly_Salary por Job_Title')
fig_3.update_layout(template='plotly_dark')

fig_4 = px.histogram(df, x='Projects_Handled', title='Histograma de Proyectos manejados', color_discrete_sequence=['blue'])
fig_4.update_layout(template='plotly_dark')

fig_5 = px.scatter(df, x='Projects_Handled', y='Monthly_Salary', color='Monthly_Salary', title='Distribución del salario por proyectos',
                   color_continuous_scale='Viridis')
fig_5.update_layout(template='plotly_dark')

# Matriz de confusión como imagen
plt.figure(figsize=(9, 9))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.title('Matriz de Confusión')

img_mat = io.BytesIO()
plt.savefig(img_mat, format='png')
img_mat.seek(0)
img_mat64 = base64.b64encode(img_mat.getvalue()).decode('utf-8')
plt.close()

# Árbol de decisión como imagen
fig_tree, ax = plt.subplots(figsize=(20, 20))
plot_tree(model, filled=True, feature_names=x.columns, class_names=model.classes_.astype(str), ax=ax)

img_tree = io.BytesIO()
plt.savefig(img_tree, format='png')
img_tree.seek(0)
img_tree64 = base64.b64encode(img_tree.getvalue()).decode('utf-8')
plt.close()

job_tlt = df['Job_Title'].unique()
job_op = [{'label': job, 'value': job} for job in job_tlt]

dcc.Dropdown(
    id='input-job-title',
    options=job_op,  # Usamos la lista generada con las opciones de los puestos
    placeholder='Selecciona un puesto de trabajo',
    style={'margin': '10px', 'width': '50%'}
)

# App Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Button("☰", id='toggle-button', style={
            'backgroundColor': '#333', 'color': 'white', 'border': 'none',
            'padding': '10px 20px', 'cursor': 'pointer', 'position': 'fixed',
            'top': '10px', 'left': '10px', 'zIndex': '1000'}),
    ]),
    html.Div(id='sidebar', children=[
        html.H2("Menú", style={'color': 'white', 'textAlign': 'center'}),
        html.Ul([
            html.Li(dcc.Link('Principal', href='/principal', style={'color': 'white', 'textDecoration': 'none'})),
            html.Li(dcc.Link('Modelo de Machine Learning', href='/ml', style={'color': 'white', 'textDecoration': 'none'})),
            html.Li(dcc.Link('Predicciones', href='/predicciones', style={'color': 'white', 'textDecoration': 'none'})),
        ], style={'listStyleType': 'none', 'padding': '0'}),
    ], style={
        'position': 'fixed',
        'width': '200px',
        'height': '100%',
        'backgroundColor': '#333',
        'padding': '20px',
        'top': '0',
        'left': '-220px',
        'transition': 'left 0.3s',
    }),
    html.Div(id='content', style={
        'marginLeft': '0',
        'padding': '20px',
        'backgroundColor': '#1A1D23',
        'color': 'white',
        'minHeight': '100vh',
        'transition': 'margin-left 0.3s',
    })
])

@app.callback(
    [Output('sidebar', 'style'),
     Output('content', 'style')],
    [Input('toggle-button', 'n_clicks')]
)
def toggle_sidebar(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        sidebar_style = {'position': 'fixed', 'width': '200px', 'height': '100%',
                         'backgroundColor': '#333', 'padding': '20px',
                         'top': '0', 'left': '0', 'transition': 'left 0.3s'}
        content_style = {'marginLeft': '220px', 'padding': '20px', 'backgroundColor': '#1A1D23',
                         'color': 'white', 'minHeight': '100vh', 'transition': 'margin-left 0.3s'}
    else:
        sidebar_style = {'position': 'fixed', 'width': '200px', 'height': '100%',
                         'backgroundColor': '#333', 'padding': '20px',
                         'top': '0', 'left': '-220px', 'transition': 'left 0.3s'}
        content_style = {'marginLeft': '0', 'padding': '20px', 'backgroundColor': '#1A1D23',
                         'color': 'white', 'minHeight': '100vh', 'transition': 'margin-left 0.3s'}
    return sidebar_style, content_style

@app.callback(
    Output('content', 'children'),
    [Input('url', 'pathname')]
)
def display_content(pathname):
    if pathname == '/ml':
        return html.Div([
            html.H1("Modelo de Machine Learning", style={'textAlign': 'center'}),
            html.P(f"Precisión del modelo: {pres:.2f}", style={'textAlign': 'center'}),
            html.Div([
                html.Img(src=f"data:image/png;base64,{img_mat64}", style={'width': '45%', 'display': 'inline-block'}),
                html.Img(src=f"data:image/png;base64,{img_tree64}", style={'width': '45%', 'display': 'inline-block'}),
            ], style={'textAlign': 'center'})
        ])
    
    elif pathname == '/predicciones':
        return html.Div([
            html.H1("Predicciones", style={'textAlign': 'center'}),
            html.Div([
                dcc.Dropdown(
                    id='input-job-title',
                    options=job_op,  # Usar las opciones generadas
                    placeholder='Selecciona un puesto de trabajo',
                    style={'margin': '10px', 'width': '50%' , 'color':'black'}
                ),
                dcc.Input(id='input-monthly-salary', type='number', placeholder='Salario mensual', style={'margin': '10px', 'width': '50%'}),
                html.Button('Predecir', id='predict-button', n_clicks=0, style={'margin': '10px'}),
            ], style={'textAlign': 'center'}),
            html.Div(id='prediction-output', style={'marginTop': '20px', 'textAlign': 'center'}),
        ])
    
    else:
        return html.Div([
            html.H1("Análisis de Rendimiento", style={'textAlign': 'center'}),
            html.Div([
                html.P([
                    "Alrededor del ",
                    html.Span("63%", style={'color': 'orange', 'fontWeight': 'bold'}),
                    " de los ",
                    html.Span("empleados", style={'color': 'orange', 'fontWeight': 'bold'}),
                    " presenta un ",
                    html.Span("rendimiento menor o igual a 3", style={'color': 'orange', 'fontWeight': 'bold'}),
                ], style={
                    'backgroundColor': '#2D2F33',
                    'color': 'white',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px auto',
                    'width': '80%',
                    'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'
                }),
            ]),
            html.Div([
                html.Div([dcc.Graph(id='fig_1', figure=fig_1)], style={'flex': 1, 'margin': '10px'}),
                html.Div([dcc.Graph(id='fig_2', figure=fig_2)], style={'flex': 1, 'margin': '10px'}),
            ], style={'display': 'flex'}),
            html.Div([
                html.Div([dcc.Graph(id='fig_3', figure=fig_3)], style={'flex': 1, 'margin': '10px'}),
                html.Div([dcc.Graph(id='fig_4', figure=fig_4)], style={'flex': 1, 'margin': '10px'}),
            ], style={'display': 'flex'}),
            html.Div([dcc.Graph(id='fig_5', figure=fig_5)], style={'margin': '10px'}),
        ])


@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-job-title', 'value'), 
     State('input-monthly-salary', 'value')]
)
def make_prediction(n_clicks, job_title, monthly_salary):
    if n_clicks > 0:
        if job_title is not None and monthly_salary is not None:
            try:
                # Verificar si el puesto de trabajo ingresado es válido
                if job_title not in job_tlt:
                    return html.Div([
                        html.H3("El puesto de trabajo ingresado no es reconocido.", style={'textAlign': 'center'})
                    ])
                
                # Codificar el job_title usando el LabelEncoder
                job_title_encoded = label.transform([job_title])[0]
                
                # Crear el DataFrame con las mismas columnas que en el entrenamiento
                input_data = pd.DataFrame({
                    'Job_Title': [job_title_encoded],
                    'Monthly_Salary': [monthly_salary]
                })
                
                # Asegurarse de que las columnas estén en el orden correcto
                input_data = input_data[['Job_Title', 'Monthly_Salary']]  # Ajustar si el orden es necesario

                # Realizar la predicción
                prediccion = model.predict(input_data)
                probabilidades = model.predict_proba(input_data)

                # Extraer la clase predicha y su probabilidad
                clase_predicha = int(prediccion[0])
                probabilidad_clase_predicha = probabilidades[0][clase_predicha] * 100  # Convertir a porcentaje

                # Mostrar el resultado
                return html.Div([
                    html.H3(f"Rendimiento Predicho: {clase_predicha}", style={'textAlign': 'center'}),
                ])
            except Exception as e:
                return html.Div([ 
                    html.H3(f"Error: {str(e)}", style={'textAlign': 'center'})
                ])
        else:
            return html.Div([ 
                html.H3("Por favor, ingrese todos los datos necesarios.", style={'textAlign': 'center'}), 
            ])
    return html.Div()



if __name__ == '__main__':
    app.run_server(debug=False)
