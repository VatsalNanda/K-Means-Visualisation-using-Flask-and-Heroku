from flask import Flask,render_template,request,redirect
from pycaret.clustering import plot_model, setup, create_model
import pandas as pd
from pycaret.internal.tabular import pull
import os 
import shutil

app = Flask(__name__)


def nopreprocessing():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False)
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')

def onlynorm():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False,normalize=True, normalize_method='zscore')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')

def onlypca():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False,pca = True, pca_method='linear')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')

def onlytransform():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False,transformation=True, transformation_method='yeo-johnson')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')

def everything():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False,transformation=True, transformation_method='yeo-johnson'
    ,normalize=True,normalize_method='zscore',pca = True, pca_method='linear')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')

def pca_and_norm():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False
    ,normalize=True,normalize_method='zscore',pca = True, pca_method='linear')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')


def norm_and_trans():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False,transformation=True, transformation_method='yeo-johnson'
    ,normalize=True,normalize_method='zscore')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')


def pca_and_transform():
    df = pd.read_csv('static/dataset.csv')
    pre = setup(data = df,silent = True,verbose=False,transformation=True, transformation_method='yeo-johnson'
    ,pca = True, pca_method='linear')
    model = create_model('kmeans',verbose=False)
    results = pull()
    plot_model(model, plot = 'elbow',save=True)
    os.remove('/Users/vatsalnanda/Desktop/Research Interns and Papers/k means clustering visualisation/static/Elbow.png')
    shutil.move('Elbow.png','static/')
    results.to_csv('static/results.csv')




@app.route("/",methods = ["GET","POST"])
def index():
    if request.method == 'POST':
        norm = request.form.get('norm')
        pca = request.form.get('pca')
        trans = request.form.get('trans')

        file = request.files["dataset"]
        file.save("static/dataset.csv")
        if pca is None and norm is None and trans is None:
            nopreprocessing()
        elif pca is not None and norm is not None and trans is not None : 
            everything()
        elif pca is not None and norm is not None : 
            pca_and_norm()
        elif pca is not None and trans is not None:
            pca_and_transform()
        elif norm is not None and trans is not None : 
            norm_and_trans()
        elif norm is not None : 
            onlynorm()
        elif pca is not None : 
            onlypca()
        elif trans is not None: 
            onlytransform()

        return redirect("/results")
    return render_template("index.html")


@app.route("/results")
def results():
    plot_image = os.path.join('static','Elbow.png')
    return render_template("results.html",content = {
        'elbow' : 'static//Elbow.png',
        'results' : 'static//results.csv'
    })


if __name__ == "__main__":
    app.run(debug= True)
    print(os.getcwd())
