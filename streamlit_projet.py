import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from cachetools import cached
#df = pd.read_csv(r"C:\Users\david\PariTennis\data_formatted\training_dataset_init.csv")
import os
# On recupère le dossier ou se trouve le script actuel
folder=os.path.dirname(__file__)
# On crée le chemin complet vers le fichier csv
path=os.path.join(folder,"training_dataset_init.csv")
df=pd.read_csv(path)
st.title("Réduction de dimension")
st.sidebar.title("Sommaire")
pages = ["Analyse en Composantes Principales","Interprétabilité"]
page = st.sidebar.radio("Aller vers", pages)
if page == pages[0] :
  st.write("### Chargement du dataframe")
st.dataframe(df.head(10))
# st.write(df.shape)
st.dataframe(df.describe())
if st.checkbox("Afficher les NA") :
  st.dataframe(df.isna().sum())
st.write("### Analyse en Composante Principales")
col_na=df.columns[df.isna().sum()!=0]
data=df.dropna(subset=col_na,axis=0,how='any')
target=data['winner_player1']
data=data.drop(['match_id','match_date','player1_birthdate','player2_birthdate','player1_name','player2_name','winner_player1'],axis=1)
# Séparation du jeu de donnée
from sklearn.model_selection import train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1234)
# Choix du selecteur
choix = ['Selecteur KBest', 'Selecteur Percentile']
option = st.selectbox('Choix du selecteur', choix)
st.write('Le selecteur choisi est :', option)
if option == 'Selecteur KBest':
   selec=SelectKBest()
elif option == 'Selecteur Percentile':
    display = st.radio('Quel percentile voulez-vous ?', ('20', '50','70','90'))
    if display == '20':
      selec=SelectPercentile(percentile=20)
    elif display == '50':
        selec=SelectPercentile(percentile=50)
    elif display == '70':
        selec=SelectPercentile(percentile=70)
    elif display == '90':
        selec=SelectPercentile(percentile=90)

X_train_sel=selec.fit_transform(X_train,y_train)
X_test_sel=selec.transform(X_test)
scaler=StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
scaler=StandardScaler().fit(X_train_sel)
X_train_sel_scaled=scaler.transform(X_train_sel)
X_test_sel_scaled=scaler.transform(X_test_sel)
pca=PCA()
X_train_pca=pca.fit_transform(X_train_sel)# PCA sans standardisation
# Affichage de la part de variance expliquée
# Refaire une PCA pour voir quels nombres d'axes 
# pca = PCA(n_components = .9)
pca1=PCA()
X_train_pca=pca1.fit_transform(X_train_sel_scaled)# Avec standardisation
fig = plt.figure()
sns.set_style("darkgrid")
plt.plot(pca1.explained_variance_ratio_.cumsum(),color='g',marker='o',linewidth=1.2)
plt.xlim(0,X_train_pca.shape[1]-1)
plt.xlabel("Nombre d'axes")
plt.ylabel('Part de Variance expliquée')
plt.title('Part de variance expliquée cumulée avec standardisation')
plt.show()
st.pyplot(fig)
# Faire l'ACP avec le nombre d'axes choisit précedement ici n=5 pour avoir 90% de variance expliquée d'après le 2è graphique

st.write("""#### Nuage de points""")
pca=PCA(n_components=3)
X_train_pca=pca.fit_transform(X_train_sel_scaled)
# Transformation des données de X_test_var
X_test_acp=pca.transform(X_test_sel_scaled)
fig = plt.figure()
fig = plt.figure(figsize=(20, 8))
plt.subplot(121)
plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train,cmap='Spectral')
plt.title("Nuage de points de l'ACP entre l'axe 1 et 2")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
st.pyplot(fig)

fig = plt.figure(figsize=(20, 8))
plt.subplot(122)
plt.scatter(X_train_pca[:,0],X_train_pca[:,2],c=y_train,cmap='Spectral')
plt.title("Nuage de points de l'ACP entre l'axe 1 et 3")
plt.xlabel('PCA1')
plt.ylabel('PCA3')
plt.show()
st.pyplot(fig)


fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
scatter=ax.scatter(X_train_pca[:,0],X_train_pca[:,1],X_train_pca[:,2],c=y_train, cmap='Spectral')
fig.colorbar(scatter)
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title("Vue générale du nuage de points")
st.pyplot(fig)

option1 = st.radio('Choose the option', ['Logistic','KNN','SVM','RandomForest'])
def Performance_PCA(model,color,label):
    score=[]
    if option == 'Selecteur KBest':
      selec=SelectKBest()
    elif (option == 'Selecteur Percentile')and display == '20':
      selec=SelectPercentile(percentile=20)
    elif (option == 'Selecteur Percentile')and display == '50':
      selec=SelectPercentile(percentile=50)
    elif (option == 'Selecteur Percentile')and display == '70':
      selec=SelectPercentile(percentile=70)
    elif (option == 'Selecteur Percentile')and display == '90':
      selec=SelectPercentile(percentile=90)
    X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1234)
    X_train_sel=selec.fit_transform(X_train,y_train)
    X_train_sel_scaled=scaler.transform(X_train_sel)
    for i in range(1,X_train_sel_scaled.shape[1]):
        X_train_sel=selec.fit_transform(X_train,y_train)
        X_test_sel=selec.transform(X_test)
        X_train_sel_scaled=scaler.transform(X_train_sel)
        X_test_sel_scaled=scaler.transform(X_test_sel)
        pca=PCA(n_components=i)
        X_train_pca=pca.fit_transform(X_train_sel_scaled)
        X_test_acp=pca.transform(X_test_sel_scaled)
        model.fit(X_train_pca,y_train)
        score.append(model.score(X_test_acp,y_test))
    st.markdown(
        """
        Représentation graphique du score en fonction de nombre d'axes k
        """
    )
    fig = plt.figure()
    plt.plot(score,color=color,linewidth=0.8,marker='.',label=label)
    plt.xlabel("Nombre d'axes conservés k")
    plt.ylabel('Score du modèle')
    plt.title("Performance du modèle en fonction du nombre d'axes conservés")
    plt.legend()
    st.pyplot(fig)
    st.write(model,"Nombres d'axes conservées:",np.argmax(score))
    st.write('Meilleur score:',score[np.argmax(score)])
with st.spinner("Loading..."):
  if option1 == 'Logistic':
    Performance_PCA(LogisticRegression(C=0.01778,penalty='l2',solver='lbfgs',random_state=123),color='orange',label='LogisticRegression')
  if option1 == 'SVM':
    Performance_PCA(SVC(kernel ='rbf'),label='SVM',color='red')
  if option1 == 'KNN':
    Performance_PCA(KNeighborsClassifier(n_neighbors=39),color='purple',label='KNN')
  if option1 == 'RandomForest':
    Performance_PCA(ensemble.RandomForestClassifier(max_features='sqrt',min_samples_split=2,random_state=123),color='green',label='RandomForest')

if page == pages[1] :
  st.write("## Interpretabilité: importance des features")
st.write("### Interpretabilité: importance des features")
st.write("#### Précision du modèle")
rl_opt=LogisticRegression ( max_iter=2000,random_state=22, C=0.01778279410038923, solver="lbfgs")
model_lr= rl_opt.fit(X_train_sel_scaled, y_train)

#tableau des  probabilités pour les joueurs  d'appartenir à la classe 0 ou la classe 1
probs = rl_opt.predict_proba(X_test_sel_scaled)
y_preds = np.where(probs[:,1]>0.4,1,0)
#
cm = pd.crosstab(y_test, y_preds, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

#courbe ROC  - outil très efficace pour évaluer un modèle
from sklearn.metrics import roc_curve, auc
fpr, tpr, seuils=roc_curve(y_test,probs[:,1] , pos_label=1)
roc_auc = auc(fpr, tpr)
fig = plt.figure()
plt.plot(fpr, tpr, color='purple', lw=2, label='Modèle clf (auc = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux faux positifs')
plt.ylabel('Taux vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()
st.pyplot(fig)

st.markdown("""
L’AUC ROC de notre modèle se situe bien au-dessus de celui d’un modèle non-informatif et en-dessous de celui d’un modèle parfait. Avec 76% d’AUC ROC, on en déduit que notre modèle est  assez performant.  
            """)

#Calcul de l'importance des features
df.match_date = df.match_date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))
dictionnaire = {'player1_birthdate':'datetime64',
                'player2_birthdate':'datetime64'}
#df_new = df.astype(dictionnaire)
df_new=df.copy()
# On boucle sur le dictionnaire pour convertir chaque colonne proprement
colonnes_existantes={k: v for k,v in dictionnaire.items() if k in df.columns}
for col, dtype in colonnes_existantes.items():
    if col in df_new.columns:
        #transforme les erreurs en vide (NaN) pour ne pas faire planter l'appli 
        if dtype in [int,float,'int64','float64']:
            df_new[col]=pd.to_numeric(df_new[col],errors='coerce')
        # On applique ensuite le type final
        try:
            df_new[col]=df_new[col].astype(dtype)
        except:
            continue
#création des variables year "année du match" et month "mois du match" pour faciliter plus tard le calcul de la moyenne mobile
df_new["year"] = df_new['match_date'].dt.year
df_new["month"] = df_new['match_date'].dt.month

#suppression det toute ligne ayant des valeurs manquantes
df_new = df_new.dropna()

df_new = df_new[(df.player1_plays > 6) & (df.player2_plays>6)]
print(df_new.shape)
df_new.head(2)
X = df_new[[ "player1_name", "player1_age", "player1_atprank", "player1_plays", "player1_wins", "player1_losses", "player1_elo", "player1_mean_serve_rating", 
            "player1_height", "player1_weight", "player1_oddsB365",
         "player2_name","player2_age", "player2_atprank", "player2_plays", "player2_wins", "player2_losses", "player2_elo", "player2_mean_serve_rating", 
          "player2_height", "player2_weight", "player2_oddsB365",         "match_date"]]
#Stockage la variable cible "winner_player1" dans une variable y
y = df_new.winner_player1

#Création d'un ensemble d'entraînement et d'un ensemble de tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=101)
col_to_drop = [ "match_date","player2_name","player1_name"]
X_train = X_train.drop(col_to_drop,axis=1)
X_test= X_test.drop(col_to_drop,axis=1)

#Pour les variables ayant des valeurs avec une étendue [0,1] ==> usage de la méthodes MinMax.
MinMax = MinMaxScaler()
col_to_minmax = ["player1_plays", "player1_wins", "player1_losses","player1_elo",
                       "player2_plays","player2_wins", "player2_losses", "player2_elo"]

X_train[col_to_minmax] = MinMax.fit_transform(X_train[col_to_minmax])
X_test[col_to_minmax] = MinMax.transform(X_test[col_to_minmax])

# Pour le reste des variables, nous utilisons la méthodes StandardScaler
scaler = StandardScaler()
col_to_std_scale = ["player1_age", "player1_atprank", "player1_elo", "player1_mean_serve_rating", "player1_height", "player1_weight", "player1_oddsB365",
         "player2_age", "player2_atprank", "player2_elo", "player2_mean_serve_rating",  "player2_height", "player2_weight", "player2_oddsB365",]

X_train[col_to_std_scale] = scaler.fit_transform(X_train[col_to_std_scale])
X_test[col_to_std_scale] = scaler.transform(X_test[col_to_std_scale])

X_train_scaler = pd.DataFrame(X_train)
X_test_scaler = pd.DataFrame(X_test)
rl_opt=LogisticRegression( max_iter=2000,random_state=22, C=0.01778279410038923, solver="lbfgs")
model_lr= rl_opt.fit(X_train_scaler, y_train)
coefficients=rl_opt.coef_
avg_importance=np.mean(np.abs(coefficients), axis=0)
feature_importance=pd.DataFrame({'Feature':X_train_scaler.columns, 'Importance':avg_importance})
feature_importance=feature_importance.sort_values('Importance',ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10,6))
# Calcul de l'importance des features
coefficients=rl_opt.coef_
avg_importance=np.mean(np.abs(coefficients), axis=0)
feature_importance=pd.DataFrame({'Feature':X_train_scaler.columns,'Importance':avg_importance})
feature_importance=feature_importance.sort_values('Importance',ascending=True)
fig = plt.figure()
sns.barplot(y='Feature', x='Importance',orient='h',data=feature_importance)
plt.title("Niveau d'importance des features obtenu avec la regression Logistique")
st.pyplot(fig)
st.dataframe(feature_importance.sort_values(by='Importance', ascending=False).head(8))
keep=['player1_age', 'player1_atprank',  'player1_oddsB365', 'player2_age',  'player2_plays', 'player2_wins', 'player2_losses',  'player2_oddsB365']

X_train_new=X_train_scaler[keep]
X_test_new=X_test_scaler[keep]

#Interprétabilité classique - PCA avec la selection des variables pertinentes
n = X_train_new.shape[1]
pca=PCA(n_components = 2)
data_2D=pca.fit_transform(X_train_new)
data_2D_test=pca.transform(X_test_new)

coeff = pca.components_.transpose()
xs = data_2D[:, 0]
ys = data_2D[:, 1]
scalex = 1.0/(xs.max() - xs.min())
scaley = 1.0/(ys.max() - ys.min())

rl_final=LogisticRegression ( max_iter=2000,random_state=22, C=0.01778279410038923, solver="lbfgs")
rl_final.fit(X_train_new,y_train)


#Interprétabilité SHARP
import shap
scaler = StandardScaler()
X_train_scaler = pd.DataFrame(X_train)
X_test_scaler = pd.DataFrame(X_test)
keep=['player1_age', 'player1_atprank', 'player1_elo', 'player1_weight', 'player1_oddsB365', 'player2_age', 'player2_atprank', 'player2_plays', 'player2_wins', 'player2_losses', 'player2_height', 'player2_weight', 'player2_oddsB365']
X_train_new=X_train_scaler[keep]
X_test_new=X_test_scaler[keep]
explainer = shap.LinearExplainer(rl_opt,X_train_scaler,nsamples=1000, feature_perturbation=None)
shap_values = explainer.shap_values(X_test_scaler)

print('Expected Value:', explainer.expected_value)
#Importance obtenu avec SHARP
rl_final=LogisticRegression ( max_iter=2000,random_state=22, C=0.01778279410038923, solver="lbfgs") 
rl_final.fit(X_train_new,y_train)

explainer = shap.LinearExplainer(rl_final,X_train_new,nsamples=1000, feature_perturbation=None)
shap_values = explainer.shap_values(X_test_new)
fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28,10))
plt.subplot(121)
shap.summary_plot(shap_values, X_test_new, plot_type="bar")
plt.title("Importance des features avec SHARP")
plt.subplot(122)
shap.summary_plot(shap_values, X_test_new)
plt.title("Graphe de densité des features")
st.pyplot(fig)
st.markdown("""
Les cotes, l’âge et le rang ATP des joueurs ont plus de poids lors de la prédiction du vainqueur du match (voir le graphique de densité à droite)
           """)