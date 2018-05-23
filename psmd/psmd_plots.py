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
import gocept.pseudonymize
#get hostname and set base directory
j = subprocess.Popen("hostname", stdout=subprocess.PIPE, shell=True)
(output, err) = j.communicate()
if 'imen' in output:
    base="/home/arasan/testrep/psmd"
else:
    base="/data/Team_Caspers/Arasan/BRIDGET"

#print base
#db merging goals
#there are 2 csv fle outputs frmo the pipelien
#concat them into signle df  - SOLVED - dfm
#need to merge dfm with subject databases
#depends on the analysis - if you want age prediction, visit age merge is enough , if you anaylsis by sex then merge with sex dbase
#there is a age database


#read in 1000 BRAINS subject list
dfs=pd.read_table(base+'/Visit1_Age.txt',sep='\t')
#u'Identifiers', u'Visit', u'Age'
#read in PSMD subjects list
dfm1=pd.read_csv(base+'/1000Brains_b1000-dMRI030_99xxxx_dwi_eddy/TOTAL_METRICS_Skel_header.csv',sep=' ')
dfm2=pd.read_csv(base+'/1000Brains_b1000-dMRI030_22xxxx_dwi_eddy/TOTAL_METRICS_Skel_header.csv',sep=' ')
dfm=pd.concat([dfm1,dfm2])
#please note that index does not concat above

# u'NAME', u'mean_skel_MD_LH_RH', u'sd_skel_MD_LH_RH',
       #u'Pw90S_skel_MD_LH_RH', u'mean_skel_FA_LH_RH', u'sd_skel_FA_LH_RH',
       #u'mean_skel_AD_LH_RH', u'sd_skel_AD_LH_RH', u'mean_skel_RD_LH_RH',
       #u'sd_skel_RD_LH_RH'
#merge keeps both key columns - figure out how to avoid this
#get 1000BRAINS age dbase - remove rows with visit2
dftb=pd.read_csv(base+'/Age_2018-04-05_08-58-19.csv',sep='\t')
#u'Data conflicts found', u'SDE complete', u'DDE complete', u'Validity',
       #u'SubjectID', u'Visit', u'Gender', u'Age', u'Date of Birth',
       #u'Date of Visit'
dftb_novisit2=dftb[dftb.Visit=='1. Visit']

#merge 
dfplot_tb=dfm.merge(dftb_novisit2,left_on='NAME', right_on='SubjectID',sort=True)
#u'NAME', u'mean_skel_MD_LH_RH', u'sd_skel_MD_LH_RH',
#       u'Pw90S_skel_MD_LH_RH', u'mean_skel_FA_LH_RH', u'sd_skel_FA_LH_RH',
#       u'mean_skel_AD_LH_RH', u'sd_skel_AD_LH_RH', u'mean_skel_RD_LH_RH',
#       u'sd_skel_RD_LH_RH', u'Data conflicts found', u'SDE complete',
#       u'DDE complete', u'Validity', u'SubjectID', u'Visit', u'Gender', u'Age',
#       u'Date of Birth', u'Date of Visit'

#dfplot_tb_final=
df_plot_tb=dfplot_tb.drop('SubjectID',axis=1)
df_plot_tb_no_visit=df_plot_tb.drop('Visit',axis=1)
df_plot_tb_no_visit_no_name=df_plot_tb_no_visit.drop('NAME',axis=1)

df_plot=dfm.merge(dfs,left_on='NAME', right_on='Identifiers',sort=True)
df_plot=df_plot.drop('Identifiers',axis=1)
df_plot_no_visit=df_plot.drop('Visit',axis=1)
df_plot_no_visit_no_name=df_plot_no_visit.drop('NAME',axis=1)
df_plot_no_visit_no_outliers=df_plot_no_visit[df_plot_no_visit.NAME != 993754]
df_plot_no_visit_no_outliers=df_plot_no_visit_no_outliers[df_plot_no_visit_no_outliers.NAME != 995964]
outliers=[991422,992427,993754,991534,995964,229085,229611]
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
        
def make_outlier_df():
    #print "rows before outlier removal"
    #print df_plot_tb.NAME.values.size
    df_plot_tb_no_outliers=df_plot_tb
    #[u'NAME', u'mean_skel_MD_LH_RH', u'sd_skel_MD_LH_RH',
    #   u'Pw90S_skel_MD_LH_RH', u'mean_skel_FA_LH_RH', u'sd_skel_FA_LH_RH',
    #   u'mean_skel_AD_LH_RH', u'sd_skel_AD_LH_RH', u'mean_skel_RD_LH_RH',
    #   u'sd_skel_RD_LH_RH', u'Data conflicts found', u'SDE complete',
    #   u'DDE complete', u'Validity', u'Visit', u'Gender', u'Age',
    #   u'Date of Birth', u'Date of Visit']
    print "before outlier removal"
    print df_plot_tb_no_outliers.NAME.values.size
    for i in outliers:
        df_plot_tb_no_outliers=df_plot_tb_no_outliers[df_plot_tb.NAME!=i]
    print "after outlier removal"
    print df_plot_tb_no_outliers.NAME.values.size   
    #print "rows before outlier removal"
    #print df_plot_tb_no_outliers.NAME.values.size
    #print df_plot_tb_no_outliers.columns
    return df_plot_tb_no_outliers 
            
#same as prev function but use age and sex database
def psmd_outlier2(sd):
    for i in df_plot_tb_no_visit.columns:
        if 'LH' in i:
            g=sns.jointplot(x='Age',y=i,data=df_plot_tb,kind='reg')
            fig = g.fig 
            fig.suptitle('Linear Regression Parameters', fontsize=12)
            plt.show()
            print "Outliers"
            print df_plot_tb_no_visit[np.abs(df_plot_tb_no_visit[i]-df_plot_tb_no_visit[i].mean())>=(sd*df_plot_tb_no_visit[i].std())] #[['NAME',i]]

#**************************************************************************            
############PLOT HIST of 22xxxx and 99xxxx ##################### 
def hist_plot():
    df_plot=make_outlier_df()
    df_plot_22xxxx=df_plot[df_plot.NAME < 989999 ]
    df_plot_99xxxx=df_plot[df_plot.NAME > 989999 ]
    ages_22 = df_plot_22xxxx.Age.values
    ages_99 = df_plot_99xxxx.Age.values
    sns.distplot(df_plot.Age.values);
    sns.distplot(ages_22);
    sns.distplot(ages_99);
#**************************************************************************            
############MAKE METRICS FILE TO BE SENT##################### 
def total_metrics():
    df_plot=make_outlier_df()
    df_plot=df_plot.drop(['Data conflicts found','SDE complete','DDE complete','Validity','Visit','Gender','Age','Date of Birth','Date of Visit'], axis=1)
    df_plot.to_csv(base+'/jureca/TOTAL_METRICS_Skel_header.csv', index=False)
    print df_plot.columns

#**************************************************************************            
############MAIN ICV CALC##################### 
def psmd_icv():
    df_plot=make_outlier_df()
    df_plot=df_plot.drop(['mean_skel_MD_LH_RH','sd_skel_MD_LH_RH','Pw90S_skel_MD_LH_RH','mean_skel_FA_LH_RH','sd_skel_FA_LH_RH','mean_skel_AD_LH_RH','sd_skel_AD_LH_RH','mean_skel_RD_LH_RH','sd_skel_RD_LH_RH','Data conflicts found','SDE complete','DDE complete','Validity','Visit','Date of Birth','Date of Visit'], axis=1)
    df_plot.to_csv(base+'/jureca/psmd_subjects_and_partners_tensor_list.csv')
    print df_plot.columns
       
#**************************************************************************            
############MAIN SKLEARN PLOT##################### 
def skl_ftr_imp():
    df_plot=make_outlier_df()
    y=df_plot.Age
    X=df_plot.drop(['NAME','Data conflicts found','SDE complete','DDE complete','Validity','Visit','Gender','Date of Birth','Date of Visit','Age'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(X_train, y_train)
    print X_train.columns.tolist()
    print type(X_train.columns)
    print model.feature_importances_
    print type(model.feature_importances_)
    f, ax = plt.subplots(figsize=(11, 9))

    #pal = sns.color_palette("Blues_d", len(y))
    #rank = y.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
    #sns.barplot(x=data.index, y=data, palette=np.array(pal[::-1])[rank])
    g=sns.barplot(x=X_train.columns.tolist(),y=model.feature_importances_,palette="Blues_d")
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    g.set(xlabel='Feature', ylabel='Importance')
    g.set_title('Metric importance in predicting age')
    plt.show()
    #predicted=clf.predict(X_test)
    #accuracy=accuracy_score(y_test,predicted)
    #print  'Out-of-bag score estimate: '+clf.oob_score_
    #print 'Mean accuracy score: {'+accuracy
#**************************************************************************            
############TEST SKLEARN PLOT##################### 
#1) to see if the outlier remoal had any effect and 2) if effect of 22xxxx is real
    
def skl_ftr_imp2():
    df_plot=make_outlier_df()
    df_plot=df_plot[df_plot.NAME > 989999 ]
    print df_plot.columns
    y=df_plot.Age
    X=df_plot.drop(['NAME','Data conflicts found','SDE complete','DDE complete','Validity','Visit','Gender','Date of Birth','Date of Visit','Age'], axis=1)
    print df_plot.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(X_train, y_train)
    print X_train.columns.tolist()
    print type(X_train.columns)
    print model.feature_importances_
    print type(model.feature_importances_)
    f, ax = plt.subplots(figsize=(11, 9))

    #pal = sns.color_palette("Blues_d", len(y))
    #rank = y.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
    #sns.barplot(x=data.index, y=data, palette=np.array(pal[::-1])[rank])
    g=sns.barplot(x=X_train.columns.tolist(),y=model.feature_importances_,palette="Blues_d")
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    g.set(xlabel='Feature', ylabel='Importance')
    g.set_title('Metric importance in predicting age')
    plt.show()
            
#**************************************************************************            
############MAIN R2 PLOT#####################      
def get_age_metric2(metric):
    df_plot=make_outlier_df()
    x=np.array([])
    y=np.array([])
    for i in df_plot.NAME:
        #if df_plot[df_plot.NAME==i]['NAME'].values.shape[0]>0:
            x=np.append(x,df_plot[df_plot.NAME==i]['Age'].values[0])
            y=np.append(y,df_plot[df_plot.NAME==i][metric].values[0])
    return x,y

def scp_linreg_metric2(metric):
    df_plot=make_outlier_df()
    x=df_plot.Age.values
    y=df_plot[metric].values
    #below is a more rigorus test of the regression variables extraction and it gave same results 
    #as above ie directly reading from df
    #x,y=get_age_metric2(metric)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, r_value, r_value**2, p_value

def psmd_r2_plots():    
    a=[]
    y=np.array([])
    #plot R^2 values
    for i in df_plot.columns:
        if 'LH' in i:
            s,_,r2,_=scp_linreg_metric2(i)
            a.append(i)
            y=np.append(y,r2)
    f, ax = plt.subplots(figsize=(11, 9))
    g=sns.barplot(x=a,y=y,palette="Blues_d")
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    g.set_title('R^2 value of metrics')
    plt.show()


#**************************************************************************            
############MAIN CORR PLOT#####################            
def psmd_corrplot():
    #df.drop(['B', 'C'], axis=1)
    dfplot=make_outlier_df()
    dfplot=dfplot.drop(['NAME','Data conflicts found','SDE complete','DDE complete','Validity','Visit','Gender','Date of Birth','Date of Visit'], axis=1)
    corr=dfplot.corr()
    #Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    #Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    #Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,annot=True, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title('Metrics - Correlation Matrix')
    plt.show()

#**************************************************************************            
############MAIN PSMD PLOT#####################
def psmd_jointplot():
    dfplot=make_outlier_df()
    for i in dfplot.columns:
        if 'LH' in i:
            g=sns.jointplot(x='Age',y=i,data=dfplot,kind='reg')
            fig = g.fig
            fig.suptitle('Linear Regression Parameters', fontsize=12)
            plt.show()

def plot_all_metrics_final(sd):
    for i in df_plot_no_visit.columns:
        if 'LH' in i:
            g=sns.jointplot(x='Age',y=i,data=df_plot_no_visit_no_outliers,kind='reg')
            fig = g.fig 
            fig.suptitle('Linear Regression Parameters', fontsize=12)
            plt.show()
            print "Outliers"
            print df_plot_no_visit_no_outliers[np.abs(df_plot_no_visit_no_outliers[i]-df_plot_no_visit_no_outliers[i].mean())>=(sd*df_plot_no_visit_no_outliers[i].std())] [['NAME',i]]


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
        
def missing_subs3():
    for i in dfm.NAME:
        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]<1:
            print "subject "+str(i)+" is not present in Visit1_Age.txt"

def missing_subs_4():
    for i in dfm.NAME:
        if dftb[dftb.SubjectID==i]['SubjectID'].values.shape[0]<1:
            print "subject "+str(i)+" is not present in Age_2018-04-05_08-58-19.csv"
            
def get_age_metric(metric):
    x=np.array([])
    y=np.array([])
    for i in dfm.NAME:
        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]>0:
            x=np.append(x,dfs[dfs.Identifiers==i]['Age'].values[0])
            print dfs[dfs.Identifiers==i]['Age'].values[0]
            y=np.append(y,dfm[dfm.NAME==i][metric].values[0])
            print dfm[dfm.NAME==i]['NAME'].values[0]
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
    print x
    print len(x)
    print y
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
def psdnmyz_2():
    #load TWO csv to be sent to be pseudonymz
    #metrics_df=pd.read_csv('/home/arasan/testrep/psmd/jureca/TOTAL_METRICS_Skel_header.csv')
    seg_df=pd.read_csv('/home/arasan/testrep/psmd/jureca/psmd_seg_vols.csv')
    #add rnadom id column to both df
    #below line is a disaster
    #metrics_df['RNDNAME'] = metrics_df['NAME'].apply(lambda x: gocept.pseudonymize.integer(x, 'secret'))
    #seg_df['RNDNAME'] = seg_df['NAME'].apply(lambda x: gocept.pseudonymize.integer(x, 'secret'))
#    a=np.random.randint(100000,999999,metrics_df.NAME.values.size)
#    metrics_df['RNDNAME']=a
#    print 'after rqndom id has been added'
#    flagg=True
#    while(flagg):
#        try:
#            print pd.concat(g for _, g in metrics_df.groupby("RNDNAME") if len(g) > 1)
#        except ValueError:
#            print 'NO DUPLICAtes'
#            metrics_df.to_csv('/home/arasan/testrep/psmd/jureca/TOTAL_rnd_temp.csv')
#            flagg=False
#        else:
#            print 'DUPES'
#            metrics_df=metrics_df.drop('RNDNAME', axis=1)
#            a=np.random.randint(100000,999999,metrics_df.NAME.values.size)
#            metrics_df['RNDNAME']=a
    #load double chekced randomeized df 1) above try catch 2) using np unique
    metrnd=pd.read_csv('/home/arasan/testrep/psmd/jureca/TOTAL_rnd_temp.csv')
    seg_df['SNO']=seg_df.index+1
    metrnd['SNO']=seg_df.index+1
    #add RNDAME column to seg_df
    seg_df['RNDNAME']=metrnd.RNDNAME.values
    #rename columns NANME to ID and RNDNAME to NAME
    seg_df=seg_df.rename(index=str, columns={"NAME": "ID"})
    seg_df=seg_df.rename(index=str, columns={"RNDNAME": "NAME"})
    metrnd=metrnd.rename(index=str, columns={"NAME": "ID"})
    metrnd=metrnd.rename(index=str, columns={"RNDNAME": "NAME"})
    #dump map out with 3 columns ID,NAME,SNO
    mapdf=metrnd[['ID','NAME','SNO']]
    mapdf.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psdnmyz_map.csv',index=False)
    #drop ID and SNO
    seg_df=seg_df.drop(['ID','SNO'],axis=1)
    metrnd=metrnd.drop(['ID','SNO'],axis=1)
    #move NAME column to first position
    metrnd=metrnd[['NAME','mean_skel_MD_LH_RH','sd_skel_MD_LH_RH','Pw90S_skel_MD_LH_RH','mean_skel_FA_LH_RH','sd_skel_FA_LH_RH','mean_skel_AD_LH_RH','sd_skel_AD_LH_RH','mean_skel_RD_LH_RH','sd_skel_RD_LH_RH']]
    seg_df=seg_df[['NAME','AGE','SEX','GMV','WMV','CSFV','ICV']]
    #if pd.concat(g for _, g in metrics_df.groupby("RNDNAME") if len(g) > 1).RNDNAME.values.size:
    #    print 'NOT OK'
    #else:
    #    print 'OK'
    metrnd.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/TOTAL_METRICS_Skel_header.csv',index=False)
    seg_df.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psmd_seg_vols.csv',index=False)
    
def psdnmyz_2_check():
    met=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/TOTAL_METRICS_Skel_header.csv')
    seg=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psmd_seg_vols.csv')
    mapp=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psdnmyz_map.csv')
    met_old=pd.read_csv('/home/arasan/testrep/psmd/jureca/TOTAL_METRICS_Skel_header.csv')
    seg_old=pd.read_csv('/home/arasan/testrep/psmd/jureca/psmd_seg_vols.csv')
    fs=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psmd_FS_vols.csv')
    fs_old=pd.read_csv('/home/arasan/testrep/psmd/jureca/eTIV_1000BRAINS_FS53.txt',delimiter='\t')
    
    #check if metrics and seg dfs have corresp values for given sbuject hence checking if the map is acurat
    cnt_match=0
    for i in met.NAME:
        #fetch ID from map df for subject
        idd=mapp[mapp.NAME==i]['ID'].values[0]
        #for every subject compare one metric from current csv and old csv
        if (met[met.NAME==i]['mean_skel_MD_LH_RH'].values[0] == met_old[met_old.NAME==idd]['mean_skel_MD_LH_RH'].values[0] and seg[seg.NAME==i]['ICV'].values[0] == seg_old[seg_old.NAME==idd]['ICV'].values[0]):
            
            cnt_match = cnt_match + 1
        else:
            print i
    print cnt_match    
    cnt_match=0
    for i in fs.NAME:
        idd=mapp[mapp.NAME==i]['ID'].values[0]
        if fs[fs.NAME==i]['EstimatedTotalIntracranialVolume'].values[0] == fs_old[fs_old.ID==idd]['EstimatedTotalIntracranialVolume'].values[0]:
            cnt_match = cnt_match + 1
        else:
            print i
    print cnt_match
        
        
    #check if map and metrics haave corresp
    #check if metrics and old metrics have corresp

def psdnmyz():
    #load TWO csv to be sent to be pseudonymz
    metrics_df=pd.read_csv('/home/arasan/testrep/psmd/jureca/TOTAL_METRICS_Skel_header.csv')
    seg_df=pd.read_csv('/home/arasan/testrep/psmd/jureca/psmd_seg_vols.csv')
    #add rnadom id column to both df
    metrics_df['RNDNAME'] = metrics_df['NAME'].apply(lambda x: gocept.pseudonymize.integer(x, 'secret'))
    #seg_df['RNDNAME'] = seg_df['NAME'].apply(lambda x: gocept.pseudonymize.integer(x, 'secret'))
    print 'after rqndom id has been added'
    print pd.concat(g for _, g in metrics_df.groupby("NAME") if len(g) > 1)
    
    #add serial nmber column
    seg_df['SNO']=seg_df.index+1
    metrics_df['SNO']=seg_df.index+1
    print 'after serial no column'
    print metrics_df.head(n=5)
    #rename original NAME column to ID, RNDNAME column to NAME
    seg_df=seg_df.rename(index=str, columns={"NAME": "ID"})
    seg_df=seg_df.rename(index=str, columns={"RNDNAME": "NAME"})
    metrics_df=metrics_df.rename(index=str, columns={"NAME": "ID"})
    metrics_df=metrics_df.rename(index=str, columns={"RNDNAME": "NAME"})
    print 'after renaming NAME to ID and RNDNAME to NAME'
    print metrics_df.head(n=5)    
    #dump map out with 3 columns ID,NAME,SNO
    mapdf=metrics_df[['ID','NAME','SNO']]
    print 'this is the map'
    print mapdf.head(n=5)
    ###mapdf.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet/psdnmyz_map.csv')
    #drop ID and SNO
    seg_df=seg_df.drop(['ID','SNO'],axis=1)
    metrics_df=metrics_df.drop(['ID','SNO'],axis=1)
    print 'after droppping ID and SNO'
    print metrics_df.head(n=5)
    #move NAME column to first position
    print metrics_df.columns.values
    print seg_df.columns.values
    metrics_df=metrics_df[['NAME','mean_skel_MD_LH_RH','sd_skel_MD_LH_RH','Pw90S_skel_MD_LH_RH','mean_skel_FA_LH_RH','sd_skel_FA_LH_RH','mean_skel_AD_LH_RH','sd_skel_AD_LH_RH','mean_skel_RD_LH_RH','sd_skel_RD_LH_RH']]
    seg_df=seg_df[['NAME','AGE','SEX','GMV','WMV','CSFV','ICV']]
    print 'shiftnig NAME to nmber 1'
    print metrics_df.head(n=5)
    ###metrics_df.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet/TOTAL_METRICS_Skel_header.csv',index=False)
    ###seg_df.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet/psmd_seg_vols.csv',index=False)
    #dump out final 2 csvs
    #print indf.columns
    #print indf[fld,fld+'2']

def psdnmyz_rnd_check():
    mapdf=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet/psdnmyz_map.csv')
    print mapdf[mapdf.duplicated(keep=False)]
    print pd.concat(g for _, g in mapdf.groupby("NAME") if len(g) > 1)

def psdnmyz_check(df):
    mapdf=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet/psdnmyz_map.csv')
    cnt=0
    for i in df.NAME:
        if df[df.NAME==i]['ID'].values[0]==mapdf[mapdf.NAME==i]['ID'].values[0]:
            cnt=cnt+1
        else:
            print df[df.NAME==i]['ID'].values[0]

def psdnmyz2():
    #fs_df - load NEW csv to be sent to be pseudonymzed
    #map_df - load map of random to original
    #fs_df
    cnt = 0
    rndid=[]
    map_df=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet/psdnmyz_map.csv')
    segdf=pd.read_csv('/home/arasan/testrep/psmd/jureca/IDs')
    for i in segdf.ID:
        if map_df[map_df.ID==i]['NAME'].values.size>0:
            #print map_df[map_df.ID==i]['NAME'].values[0]
            #rndid.append(map_df[map_df.ID==i]['NAME'].values[0])
            cnt = cnt +1
        else:
            print 'missing '+str(i)
            
    print cnt
        #rndid.append(map_df[map_df.ID==i]['NAME'].values[0])
    #segdf=segdf.rename(index=str, columns={"NAME": "ID"})
    #segdf['NAME']=rndid
    #segdf.to_csv('/home/arasan/testrep/psmd/jureca/IDs_pseudonymized.csv')
    
#pseudonymisze FS data sent by Bittner on 5/18
def psdnmyz3():
    fs2=pd.read_csv('/home/arasan/testrep/psmd/jureca/eTIV_1000BRAINS_FS53.txt',delimiter='\t')
    #REMOVE OUTLIERS
    fs2_no_outliers=fs2
    print 'nsubj before outlier removal'
    print fs2_no_outliers.ID.values.size 
    for i in outliers:
        fs2_no_outliers=fs2_no_outliers[fs2_no_outliers.ID!=i]
    print 'nsubj afteer outlier removal'
    print fs2_no_outliers.ID.values.size
    #REMOVE SUBJ WITHOUT FSL ICV
    seg=pd.read_csv('/home/arasan/testrep/psmd/jureca/psmd_seg_vols.csv')
    rmvlist=[]
    for i in fs2_no_outliers.ID:
        if seg[seg.NAME==i]['NAME'].values.shape[0]<1:
            print "subject "+str(i)+" in fs2_no_out is not present in seg"
            rmvlist.append(i)
    # remove rows from fs2_no_outliers that are no prewent in seg - because they have no fsl ICV
    fs2_no_outliers_no_fslmissing=fs2_no_outliers
    for i in rmvlist:
        fs2_no_outliers_no_fslmissing=fs2_no_outliers_no_fslmissing[fs2_no_outliers_no_fslmissing.ID!=i]
    print 'nsubj afteer fslmissing removal - final to be sent to BRODEAUX for FS ICV'
    print fs2_no_outliers_no_fslmissing.ID.values.size   
    
    #for i in seg.NAME:
    #    if fs2_no_outliers[fs2_no_outliers.ID==i]['ID'].values.shape[0]<1:
    #        print "subject "+str(i)+" present in seg is not present in fs2_no_out"
    #print fs2_no_outliers
    #LOAD MAP
    map_df=pd.read_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psdnmyz_map.csv')
    cnt=0
    rndid=[]
    for i in fs2_no_outliers_no_fslmissing.ID:
        if map_df[map_df.ID==i]['NAME'].values.size>0:
            #print map_df[map_df.ID==i]['NAME'].values[0]
            rndid.append(map_df[map_df.ID==i]['NAME'].values[0])
            cnt = cnt +1
        else:
            print 'missing subj in MAP'+str(i)
    #print rndid
    #print cnt
    print 'length of list reterived from MAP - should match nsubj in fs2_no_outliers_no_fslmissing'
    print len(rndid)
    #CREATE NW COLUMN FROM RNDID - CALL IT NAME
    fs2_no_outliers_no_fslmissing['NAME']=rndid
    #fs2_no_outliers['NAME']=pd.Series(rndid).values
    #print fs2_no_outliers_no_fslmissing
    
    cnt=0
    #print fs2_no_outliers_no_fslmissing
    for i in fs2_no_outliers_no_fslmissing.NAME:
        if fs2_no_outliers_no_fslmissing[fs2_no_outliers_no_fslmissing.NAME==i]['ID'].values[0]==map_df[map_df.NAME==i]['ID'].values[0]:
            cnt=cnt+1
        else:
            print 'WRONG MAP'
            print i
            print fs2_no_outliers_no_fslmissing[fs2_no_outliers_no_fslmissing.NAME==i]['ID'].values[0]
            print map_df[map_df.NAME==i]['ID'].values[0]
    print 'number of correct matches should match nsubj after fslmissing removal'
    print cnt    
    
    #drop ID
    fs2_no_outliers_no_fslmissing=fs2_no_outliers_no_fslmissing.drop(['ID'],axis=1)
    
    #move NAME to first position
    fs2_no_outliers_no_fslmissing=fs2_no_outliers_no_fslmissing[['NAME','EstimatedTotalIntracranialVolume']]
    print fs2_no_outliers_no_fslmissing

    fs2_no_outliers_no_fslmissing.to_csv('/home/arasan/testrep/psmd/jureca/bordeaux_packet2/psmd_FS_vols.csv',index=False)
    
    
    #fs2_no_outliers.drop('ID', axis=1)
    #segdf.to_csv('/home/arasan/testrep/psmd/jureca/IDs_pseudonymized.csv')
    
    