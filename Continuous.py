


def diagnostic_plots():
    pass
    feature_importance()
    residuals_vs_fitted()
    
    


# predictions_df['Residuals'], predictions_df['Prediction']

#Residuals vs Fitted Plot
def residuals_vs_fitted(residuals, prediction):
    smoothed = lowess(residuals,prediction)
    top3 = abs(residuals).sort_values(ascending = False)[:3]

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,7)
    fig, ax = plt.subplots()
    ax.scatter(prediction, residuals, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Residuals vs. Fitted')
    ax.plot([min(prediction),max(prediction)],[0,0],color = 'k',linestyle = ':', alpha = .3)

    #Annotate the max and min values
    for i in top3.index:
        ax.annotate(i,xy=(predictions_df['Prediction'][i],predictions_df['Residuals'][i]))
    
    
    # In this case, you only learn what the predictions look like to potentially see outliers, or trends.
    # Heteroskedasticity is not an assumption to check
    # Does not show how a particular observation fell in the tree.


# Variable Importance Bar Graph
def feature_importance():
    feature_importance=pd.DataFrame({'xgboost_model':cv_best_model.feature_importances_},index=X_train.columns)
    feature_importance.sort_values(by='xgboost_model',ascending=True,inplace=True)

    index = np.arange(len(feature_importance))
    fig, ax = plt.subplots(figsize=(12,8))
    rfc_feature=ax.barh(index,feature_importance['xgboost_model'],0.4,color='dodgerblue',label='XGBoost Model')
    ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

    ax.legend()
    plt.show()
