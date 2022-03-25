library(pROC)
library(e1071)
library(dplyr)
library(factoextra)
library(ggplot2)
library(glmnet)
library(caret)
library(FNN)
library(kknn)
#get train data
#######################################
creditcard.df <- read.csv("analyze(2).csv")
creditcard.df <- creditcard.df[with(creditcard.df, order(X)),]
##delete $ in Amount
creditcard.df$Amount<-as.numeric(gsub('[$]','',creditcard.df$Amount))

##Add a new column which is the hour that the transaction happened
creditcard.df$Hour <- substring(creditcard.df$Time,1,2)
summary(creditcard.df)

#Divided a day (24 hours) into three periods: Period 1 (21-4), Period 2 (5-12), Period 3 (13-20)
creditcard.df$Night <- ifelse(creditcard.df$Hour %in% c('00','01','02','03','04','21','22','23'),1,0)
creditcard.df$Morning <- ifelse(creditcard.df$Hour %in% c('08','09','10','11','12','05','06','07'),1,0)
creditcard.df$Afternoon <- ifelse(creditcard.df$Hour %in% c('16','17','18','19','20','13','14','15'),1,0)

#Transfer the following variables into dummy variables
creditcard.df$FraudOrNot[creditcard.df$Is.Fraud. == "Yes"] <- 1
creditcard.df$FraudOrNot[creditcard.df$Is.Fraud. =="No"] <- 0
creditcard.df$Chip[creditcard.df$Use.Chip == "Chip Transaction"] <- 1
creditcard.df$Chip[creditcard.df$Use.Chip != "Chip Transaction"] <- 0

creditcard.df$Swipe[creditcard.df$Use.Chip == "Swipe Transaction"] <- 1
creditcard.df$Swipe[creditcard.df$Use.Chip != "Swipe Transaction"] <- 0

creditcard.df$Online[creditcard.df$Use.Chip == "Online Transaction"] <- 1
creditcard.df$Online[creditcard.df$Use.Chip != "Online Transaction"] <- 0

#Categorize the error types into 7 types of transaction errors, and transform them into dummy variables
creditcard.df$TechGlitch <- ifelse(creditcard.df$Errors. %in% c("Technical Glitch","Bad PIN,Technical Glitch","Insufficient Balance,Technical Glitch","Bad Card Number,Technical Glitch","Bad Zipcode,Technical Glitch"),1,0)
creditcard.df$BadCardNum <- ifelse(creditcard.df$Errors. %in% c("Bad Card Number","Bad Card Number,Technical Glitch"),1,0)
creditcard.df$InsufBalance <- ifelse(creditcard.df$Errors. %in% c("Insufficient Balance","Insufficient Balance,Technical Glitch","Bad PIN,Insufficient Balance"),1,0)
creditcard.df$BadPin <- ifelse(creditcard.df$Errors. %in% c("Bad PIN","Bad PIN,Technical Glitch","Bad PIN,Insufficient Balance"),1,0)
creditcard.df$BadExpire <- ifelse(creditcard.df$Errors. %in% c("Bad Expiration","Bad Expiration,Bad CVV"),1,0)
creditcard.df$BadCVV <- ifelse(creditcard.df$Errors. %in% c("Bad CVV","Bad Expiration,Bad CVV"),1,0)
creditcard.df$BadZip <- ifelse(creditcard.df$Errors. %in% c("Bad Zipcode","Bad Zipcode,Technical Glitch"),1,0)

creditcard.df <- creditcard.df[,-1]
train <- creditcard.df
df1 <- creditcard.df

#get test data
#######################################
creditcard.df <- read.csv("test.csv")
creditcard.df <- creditcard.df[with(creditcard.df, order(X)),]
##delete $ in Amount
creditcard.df$Amount<-as.numeric(gsub('[$]','',creditcard.df$Amount))

#Add a new column which is the hour that the transaction happened
creditcard.df$Hour <- substring(creditcard.df$Time,1,2)
summary(creditcard.df)

#Divided a day (24 hours) into three periods: Period 1 (21-4), Period 2 (5-12), Period 3 (13-20)
creditcard.df$Night <- ifelse(creditcard.df$Hour %in% c('00','01','02','03','04','21','22','23'),1,0)
creditcard.df$Morning <- ifelse(creditcard.df$Hour %in% c('08','09','10','11','12','05','06','07'),1,0)
creditcard.df$Afternoon <- ifelse(creditcard.df$Hour %in% c('16','17','18','19','20','13','14','15'),1,0)

#Transfer the following variables into dummy variables
creditcard.df$FraudOrNot[creditcard.df$Is.Fraud. == "Yes"] <- 1
creditcard.df$FraudOrNot[creditcard.df$Is.Fraud. =="No"] <- 0
creditcard.df$Chip[creditcard.df$Use.Chip == "Chip Transaction"] <- 1
creditcard.df$Chip[creditcard.df$Use.Chip != "Chip Transaction"] <- 0

creditcard.df$Swipe[creditcard.df$Use.Chip == "Swipe Transaction"] <- 1
creditcard.df$Swipe[creditcard.df$Use.Chip != "Swipe Transaction"] <- 0

creditcard.df$Online[creditcard.df$Use.Chip == "Online Transaction"] <- 1
creditcard.df$Online[creditcard.df$Use.Chip != "Online Transaction"] <- 0

#Categorize the error types into 7 types of transaction errors, and transform them into dummy variables
creditcard.df$TechGlitch <- ifelse(creditcard.df$Errors. %in% c("Technical Glitch","Bad PIN,Technical Glitch","Insufficient Balance,Technical Glitch","Bad Card Number,Technical Glitch","Bad Zipcode,Technical Glitch"),1,0)
creditcard.df$BadCardNum <- ifelse(creditcard.df$Errors. %in% c("Bad Card Number","Bad Card Number,Technical Glitch"),1,0)
creditcard.df$InsufBalance <- ifelse(creditcard.df$Errors. %in% c("Insufficient Balance","Insufficient Balance,Technical Glitch","Bad PIN,Insufficient Balance"),1,0)
creditcard.df$BadPin <- ifelse(creditcard.df$Errors. %in% c("Bad PIN","Bad PIN,Technical Glitch","Bad PIN,Insufficient Balance"),1,0)
creditcard.df$BadExpire <- ifelse(creditcard.df$Errors. %in% c("Bad Expiration","Bad Expiration,Bad CVV"),1,0)
creditcard.df$BadCVV <- ifelse(creditcard.df$Errors. %in% c("Bad CVV","Bad Expiration,Bad CVV"),1,0)
creditcard.df$BadZip <- ifelse(creditcard.df$Errors. %in% c("Bad Zipcode","Bad Zipcode,Technical Glitch"),1,0)
creditcard.df <- creditcard.df[,-1]
test <- creditcard.df
train <- select(train, -1:-6,-8:-16)
test <- select(test, -1:-6,-8:-16)


#logistic regression
y_train <- train$FraudOrNot
y_test <- test$FraudOrNot
x_train <- select(train, -5)
x_test <- select(test, -5)

set.seed(1)
xtrain<- as.matrix(x_train)
xtest <- as.matrix(x_test)
fit_cv <- cv.glmnet(xtrain, y_train, alpha=1, family = 'binomial', type.measure='auc')
plot(fit_cv)
minlambda <- fit_cv$lambda.min
lasso_model <- glmnet(xtrain, y_train, alpha = 1, lambda = minlambda,  family = 'binomial',standardize = TRUE)
pred_train <- predict(lasso_model,xtrain,type = "response")
pred_test <- predict(lasso_model,xtest,type = "response")

#Draw the ROC curve
lr_roc <- roc(y_test,pred_test)
plot(lr_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='Lasso-Logistic Regression')
#AUC=0.832 enough to explain

#evaluate the model from business perspectives
result_review_lr<- data.frame(amount = x_test[,1], obs = y_test, pred = pred_test)

result_review_lr$s0 <- ifelse(result_review_lr$s0<0.503,0,1)
names(result_review_lr)[names(result_review_lr) == 's0'] <- 'pred'
head(result_review_lr)
table(y_test,result_review_lr$pred,dnn=c("Real","Predict"))

# confusion_matrix
confusion_matrix <- as.data.frame(table(y_test,result_review_lr$pred,dnn=c("Real","Predict")))
confusion_matrix
ggplot(data = confusion_matrix,
       mapping = aes(x = Real,
                     y = Predict)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white",high = "gray",trans = "log")+
  ggtitle("Logistic Regression Confusion Matrix") +  theme(plot.title = element_text(hjust = 0.5))

#Calculate fraud detection rate
TF_amount <- sum(result_review_lr$amount*result_review_lr$obs*result_review_lr$pred)
Fraud_amount<- sum(result_review_lr$amount*result_review_lr$obs)
Fraud_det_rate <- TF_amount/Fraud_amount
print(c("TF_amount=",TF_amount,"Fraud_amount=",Fraud_amount,"Fraud_det_rate=",Fraud_det_rate))



#svm-classification
x_svm<- svm(FraudOrNot~ .,data=train,
            type = 'C',kernel = 'radial' )
pre_svm <- predict(x_svm,newdata = x_test)
length(pre_svm)
obs_p_svm = data.frame(prob=pre_svm,obs=y_test)

#Draw the ROC curve
table(y_test,pre_svm,dnn=c("Real","Predict"))
class(y_test)
pre_svm_num <- as.numeric(pre_svm)
svm_roc <- roc(y_test,pre_svm_num)
plot(svm_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM X ROC kernel = radial')

#evaluate the model from business perspectives
pre_svm <- as.numeric(pre_svm)
obs_p_svm = data.frame(amount = x_test[,1],prob=pre_svm,obs=y_test)
obs_p_svm$prob<- ifelse(obs_p_svm$prob<1.5,0,1)
head(obs_p_svm)

# confusion_matrix
confusion_matrix <- as.data.frame(table(y_test,obs_p_svm$pred,dnn=c("Real","Predict")))
confusion_matrix
ggplot(data = confusion_matrix,
       mapping = aes(x = Real,
                     y = Predict)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white",high = "gray",trans = "log")+
  ggtitle("SVM Confusion Matrix") +  theme(plot.title = element_text(hjust = 0.5))

#Calculate fraud detection rate
TF_amount_svm <- sum(obs_p_svm$amount*obs_p_svm$obs*obs_p_svm$prob)
Fraud_amount_svm<- sum(obs_p_svm$amount*obs_p_svm$obs)
Fraud_det_rate_svm <- TF_amount_svm/Fraud_amount_svm
print(c("TF_amount_svm=",TF_amount_svm,"Fraud_amount_svm=",Fraud_amount_svm,"Fraud_det_rate_svm=",Fraud_det_rate_svm))

#knn
creditcard.df2 <- select(df1, -1:-6,-8:-16)
creditcard.df2 <- select(creditcard.df2, -5)

train.norm.df <- train
valid.norm.df <- test
creditcard.norm.df <- creditcard.df2

norm.values <- preProcess(creditcard.df2, method=c("center", "scale"))
train.norm.df <- predict(norm.values, train)
valid.norm.df <- predict(norm.values, test)
creditcard.norm.df <- predict(norm.values, creditcard.df)

bank.knn <- knn(train = train.norm.df[ , -c(5)], test = valid.norm.df[ , -c(5)] , cl = train.norm.df[, 5], k = 1)
summary(bank.knn)

#choose the best k
accuracy.df <- data.frame(k =  seq(1, 14, 1), accuracy = rep(0, 14))

for(i in 1:14) {
  credit.knn.pred <- knn(train.norm.df[, -c(5)], valid.norm.df[, -c(5)], cl = train.norm.df[, 5], k=i)
  accuracy.df[i, 2] <- confusionMatrix(as.factor(credit.knn.pred), as.factor(valid.norm.df[, c(5)]))$overall[1]
}
accuracy.df

# k = 2 is the best
knn2 <- kknn(FraudOrNot~ .,
              train.norm.df,valid.norm.df,k=2,distance = 2)
pre_knn2 <- fitted(knn2)

#Draw the ROC curve
knn2_roc <- roc(valid.norm.df[ , 5],as.numeric(pre_knn2))
plot(knn2_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"),
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='knn2 X ROC')

#evaluate the model from business perspectives
pre_knn2 <- as.numeric(pre_knn2 )
obs_p_knn2  = data.frame(amount = x_test[,1],prob=pre_knn2,obs=y_test)
obs_p_knn2$prob<- ifelse(obs_p_knn2$prob<0.5,0,1)
head(obs_p_knn2)
table(y_test,obs_p_knn2$prob,dnn=c("Real","Predict"))

# confusion_matrix
confusion_matrix <- as.data.frame(table(y_test,obs_p_knn2$pred,dnn=c("Real","Predict")))
confusion_matrix
ggplot(data = confusion_matrix,
       mapping = aes(x = Real,
                     y = Predict)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white",high = "gray",trans = "log")+
  ggtitle("KNN Confusion Matrix") +  theme(plot.title = element_text(hjust = 0.5))

#Calculate fraud detection rate
TF_amount_knn2 <- sum(obs_p_knn2$amount*obs_p_knn2$obs*obs_p_knn2$prob)
Fraud_amount_knn2<- sum(obs_p_knn2$amount*obs_p_knn2$obs)
Fraud_det_rate_knn2 <- TF_amount_knn2/Fraud_amount_knn2
print(c("TF_amount_knn2=",TF_amount_knn2,"Fraud_amount_knn2=",Fraud_amount_knn2,"Fraud_det_rate_knn2=",Fraud_det_rate_knn2))


#PCA 
dat_eigen<-scale(x_train,scale=T)%>%cor()%>%eigen()
dat_eigen$values 
dat_eigen$vectors
sweep(dat_eigen$vectors,2,sqrt(dat_eigen$values),"*")
scale(x_train,scale=T)%*%dat_eigen$vectors%>%head() 

x.pca<-prcomp(x_train,scale=T,rank=10,retx=T)
summary(x.pca)
x.pca$sdev
x.pca$rotation
x.pca$x

fviz_eig(x.pca,addlabels = TRUE)
fviz_pca_var(x.pca)
fviz_contrib(x.pca, choice = "var", axes = 1:5)

#Draw a graph to demonstrate 
plot(x.pca,main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)
datapc <- predict(x.pca)

# Find explain variation percentage
vars <- (x.pca$sdev)^2 
vars
props <- vars / sum(vars)    
props
cumulative.props <- cumsum(props) 
cumulative.props
plot(cumulative.props)

# pca$rotation 
top4_pca.data <- x.pca$x[, 1:4]
top4_pca.data 

top4.pca.eigenvector <- x.pca$rotation[, 1:4]
top4.pca.eigenvector


first.pca <- top4.pca.eigenvector[, 1]  
second.pca <- top4.pca.eigenvector[, 2]  
third.pca <- top4.pca.eigenvector[, 3]  
fourth.pca <- top4.pca.eigenvector[, 4]  


first.pca[order(first.pca, decreasing=FALSE)] 
dotchart(first.pca[order(first.pca, decreasing=FALSE)] ,   
         main="Loading Plot for PC1",                     
         xlab="Variable Loadings",                        
         col="red")                                       


second.pca[order(second.pca, decreasing=FALSE)] 
dotchart(second.pca[order(second.pca, decreasing=FALSE)] ,   
         main="Loading Plot for PC2",                    
         xlab="Variable Loadings",                       
         col="red")                                       


third.pca[order(third.pca, decreasing=FALSE)] 
dotchart(third.pca[order(third.pca, decreasing=FALSE)] ,   
         main="Loading Plot for PC3",                     
         xlab="Variable Loadings",                      
         col="red")                                      


fourth.pca[order(fourth.pca, decreasing=FALSE)] 
dotchart(fourth.pca[order(fourth.pca, decreasing=FALSE)] ,  
         main="Loading Plot for PC4",                     
         xlab="Variable Loadings",                        
         col="red")                                      

pcaplot <- datapc[,1:4]
pcaplot <- cbind(pcaplot, fraud = y_train)
head(pcaplot)
pcaplot <- as.data.frame(pcaplot)


# biplot(pca.data, choices=1:2)  
ggplot(data=pcaplot, aes(x=PC1, y=PC2, color=fraud))+geom_point(size=1) + labs(title="PC1 vs PC2") 
ggplot(data=pcaplot, aes(x=PC3, y=PC4, color=fraud))+geom_point(size=1) + labs(title="PC3 vs PC4")

