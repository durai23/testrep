import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import subprocess
#get hostname and set base directory
j = subprocess.Popen("hostname", stdout=subprocess.PIPE, shell=True)
(output, err) = j.communicate()
if 'imen' in output:
    base="/home/arasan/testrep/psmd"
else:
    base="/data/Team_Caspers/Arasan/BRIDGET/1000Brains_b1000-dMRI030_99xxxx_dwi_eddy"

print base
#read in 1000 BRAINS subject list
dfs=pd.read_table(base+'/Visit1_Age.txt',sep='\t')
#read in PSMD subjects list
dfm=pd.read_csv(base+'/TOTAL_METRICS_Skel_header.csv',sep=' ')
#merge keeps both key columns - figure out how to avoid this
#get 1000BRAINS age dbase - remove rows with visit2
dftb=pd.read_csv(base+'/Age_2018-04-05_08-58-19.csv',sep='\t')
dftb_novisit2=dftb[dftb.Visit=='1. Visit']

#merge 
dfplot_tb=dfm.merge(dftb_novisit2,left_on='NAME', right_on='SubjectID',sort=True)
#dfplot_tb_final=
df_plot_tb=dfplot_tb.drop('SubjectID',axis=1)
df_plot_tb_no_visit=df_plot_tb.drop('Visit',axis=1)
df_plot_tb_no_visit_no_name=df_plot_tb_no_visit.drop('NAME',axis=1)
dfplot=dfm.merge(dfs,left_on='NAME', right_on='Identifiers',sort=True)
df_plot=dfplot.drop('Identifiers',axis=1)
df_plot_no_visit=df_plot.drop('Visit',axis=1)
df_plot_no_visit_no_name=df_plot_no_visit.drop('NAME',axis=1)
df_plot_no_visit_no_outliers=df_plot_no_visit[df_plot_no_visit.NAME != 993754]
df_plot_no_visit_no_outliers=df_plot_no_visit_no_outliers[df_plot_no_visit_no_outliers.NAME != 995964]
#df_plot_no_visit_no_name=dfplot.drop(['Identifiers', 'Visit','NAME'], axis=1).columns
#sns.heatmap(df_plot_no_visit.corr(),xticklabels=df_plot_no_visit.corr().columns.values,yticklabels=df_plot_no_visit.corr().columns.values)

#plot all linreg metricds while listing outliers
def psmd_outlier(sd):
    for i in df_plot_no_visit.columns:
        if 'LH' in i:
            g=sns.jointplot(x='Age',y=i,data=df_plot,kind='reg')
            fig = g.fig 
            fig.suptitle('Linear Regression Parameters', fontsize=12)
            plt.show()
            print "Outliers"
            print df_plot_no_visit[np.abs(df_plot_no_visit[i]-df_plot_no_visit[i].mean())>=(sd*df_plot_no_visit[i].std())] [['NAME',i]]

def plot_all_metrics_final(sd):
    for i in df_plot_no_visit.columns:
        if 'LH' in i:
            g=sns.jointplot(x='Age',y=i,data=df_plot_no_visit_no_outliers,kind='reg')
            fig = g.fig 
            fig.suptitle('Linear Regression Parameters', fontsize=12)
            plt.show()
            print "Outliers"
            print df_plot_no_visit_no_outliers[np.abs(df_plot_no_visit_no_outliers[i]-df_plot_no_visit_no_outliers[i].mean())>=(sd*df_plot_no_visit_no_outliers[i].std())] [['NAME',i]]

def skl_ftr_imp():
    y=df_plot_no_visit_no_name.Age
    X=df_plot_no_visit_no_name.drop('Age',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(X_train, y_train)
    print X_train.columns.tolist()
    print type(X_train.columns)
    print model.feature_importances_
    print type(model.feature_importances_)
    
    pal = sns.color_palette("Blues_d", len(y))
    rank = y.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
    #sns.barplot(x=data.index, y=data, palette=np.array(pal[::-1])[rank])
    g=sns.barplot(x=X_train.columns.tolist(),y=model.feature_importances_, color='blue')
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    #predicted=clf.predict(X_test)
    #accuracy=accuracy_score(y_test,predicted)
    print  'Out-of-bag score estimate: '+clf.oob_score_
    print 'Mean accuracy score: {'+accuracy

def dia_corr_mat():
    corr=df_plot_no_visit_no_name.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    #f, ax = plt.subplots()
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,annot=True, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
#function to list subjects in PSMD results but not in 1000Brains list
def missing_subs():
    for i in dfm.NAME:
        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]<1:
            print "subject "+str(i)+" is not present in Visit1_Age.txt"

def missing_subs_2():
    for i in dfm.NAME:
        if dftb[dftb.SubjectID==i]['SubjectID'].values.shape[0]<1:
            print "subject "+str(i)+" is not present in Age_2018-04-05_08-58-19.csv"
            
def get_age_metric(metric):
    x=np.array([])
    y=np.array([])
    for i in dfm.NAME:
        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]>0:
            x=np.append(x,dfs[dfs.Identifiers==i]['Age'].values[0])		
            y=np.append(y,dfm[dfm.NAME==i][metric].values[0])
    return x,y

#plots line reg params - metric vs age
def plot_metric(metric):
    x,y=get_age_metric(metric)
    data = pd.DataFrame(data={'Age': x, metric: y})
    print x.shape
    #grid = sns.lmplot('Age', 'MD', data, size=7, truncate=True, scatter_kws={"s": 100})
    #sns.lmplot(x='Age',y='MD', data=data)
    #plt.show()
    
    #sns.regplot(x='Age',y=metric,data=data)
    sns.jointplot(x='Age',y=metric,data=data,kind='reg')

def plot_metric_2(metric):
#    print metric
    sns.jointplot(x='Age',y=metric,data=df_plot_no_visit,kind='reg')
    plt.show()
    #print df_plot_no_visit[np.abs(df_plot_no_visit.Pw90S_skel_MD_LH_RH-df_plot_no_visit.Pw90S_skel_MD_LH_RH.mean())=>(4*df_plot_no_visit.Pw90S_skel_MD_LH_RH.std())]['NAME'].values
    print "Outliers"
    print df_plot_no_visit[np.abs(df_plot_no_visit.Pw90S_skel_MD_LH_RH-df_plot_no_visit.Pw90S_skel_MD_LH_RH.mean())>=(4*df_plot_no_visit.Pw90S_skel_MD_LH_RH.std())] [['NAME',metric]]#['NAME'].values
    
def scp_linreg_metric(metric):
    x,y=get_age_metric(metric)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, r_value, r_value**2, p_value

def scp_linreg_plot():
    a=[]
    y=np.array([])
    for i in df_plot_no_visit_no_name.columns:
#        print i
        if 'LH' in i:
            s,_,r2,_=scp_linreg_metric(i)
            a.append(i)
            y=np.append(y,r2)
    g=sns.barplot(x=a,y=y, color='blue')
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    

def sm_ols_metric(metric):
    x,y=get_age_metric(metric)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    #predictions = model.predict(x) # make the predictions by the model
    # Print out the statistics
    print model.summary()
    
def skl_linreg_metric(metric):
    x,y=get_age_metric(metric)
    x=x.reshape(-1,1)
    y=y.reshape(-1,1)
    model=LinearRegression()
    model.fit(x,y)
    print model.score(x,y)
  
def make_df_reg_all(sd=True):    
    df_list=[]
    for i in dfm.columns.values.tolist():
        if sd:
            if "LH" in i and sd:
                slope,r,r2,p=scp_linreg_metric(i)
                row_dict={'metric':i,'slope':slope,'pearsonr':r,'r-squared':r2,'p':p}
                df_list.append(row_dict)
        else:
            if "LH" in i and "sd" not in i:
                slope,r,r2,p=scp_linreg_metric(i)
                row_dict={'metric':i,'slope':slope,'pearsonr':r,'r-squared':r2,'p':p}
                df_list.append(row_dict)
            
            #print "For "+i+":"
            #print "p-value :"+str(p)
            #print "R-squared :"+str(r2)
            #print "Slope :"+str(slope)
    df=pd.DataFrame(df_list)
    print df
    df.to_csv(base+'/linreg_metrics.csv')
   
def polyfit2(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results

#def parse_metric(metric):
#    if metric=='mean_skel_MD_LH_RH':
#        return mean_MD
#    elif metric=='sd_skel_MD_LH_RH':
#        return sd_MD    .
    
#def df_corr_all():
#    df_corr_list=[]
#    x=np.array([])
#    y=np.array([])
#    for i in dfm.NAME:
#        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]>0:
#            x=np.append(x,dfs[dfs.Identifiers==i]['Age'].values[0])		
#            for j in dfm.columns.values.tolist():
#                if "LH" in j:
#                    y=np.append(y,dfm[dfm.NAME==i][j].values[0])
#
#                    
#            y=np.append(y,dfm[dfm.NAME==i][metric].values[0])
#    
#    Age,_ = get_age_metric()
#    for i in dfm.NAME:
#        if 
#        row_dict={'Age':}