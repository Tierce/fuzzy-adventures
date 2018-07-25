# Section 1 (Load Libraries and Configure Settings)===============================================================================================================
# 1.1 Load libraries
# install.packages('caret', dependencies = TRUE)
library(stringdist) #To calculate distance metrics
library(stringr) # To manipulate strings
library(reshape2) #For Dataframe manipulation
library(plyr) #For data manipulation
library(dplyr) #For data manipulation
library(textclean) #To clean text of punctuation and whitespaces
library(tm) #For text manipulation
library(xgboost) #ML model
library(caTools) #For splitting data into train/test
library(caret) #For Confusian matrix
library(MlBayesOpt) #For Bayesian Optimization of xgboost
library(purrr) #For Data manipulation
library(ggplot2) #For Plotting
starttime1 <- Sys.time()

# 1.2 Set working directory (where data and model files are stored)
setwd("C:/Users/a0764754/Desktop/R/R data folder")

# Section 2 (Upload train data, clean data, generate datasets for training) =============================================

# 2.1 Set words to remove from train data 
remove.word <- c('International', 'Incorporation',"corporation",'Company',"holdings","holding","limited",'Group','corp',"gmbh","ltd","llc","inc","pte","pvt","plc","the",
                 "bv","nv","lp","l l c","n v","b v","l l p","p l c", 'SA')

# 2.2 Import and Clean data 
data <- read.delim("Output2.csv", header=T,sep=",", stringsAsFactors = F)
data[1:2] <- lapply(data[1:2], toupper) #convert to uppercase
data[1:2] <- lapply(data[1:2],strip,digit.remove = F, lower.case = F) #minus punctuation
data[1:2] <- map_df(data[1:2], function (y) gsub("[^A-Za-z0-9]", " ", y)) #Minus non alphanumeric characters
data[1:2] <- lapply(data[1:2],trimws) #minus whitespace

# 2.3 Label Matches and save names as a placeholder
colnames(data) <- c("Var1","Var2")
data$match <- 1
placeholder <- merge(data$Var1,data$Var2,by=NULL)
colnames(placeholder) <- c("Var1","Var2")
placeholder<- join(placeholder,data,by=c("Var1","Var2"))
placeholder$match[is.na(placeholder$match)] <- 0

# 2.4 Create data to compare non-matching words within strings
df <- placeholder[1:2]
colnames(df) <- c("x","y")
df <- data.frame(lapply(df[1:2],as.character),stringsAsFactors = FALSE)
df$dupe <- apply(df,1,function(x)
  paste(Reduce(intersect, strsplit(x, " ")), collapse = " ")) #Matching words
df$dupe[df$dupe==""] <- NA
dupes <- strsplit(df$dupe," ")
df$dupe <- lapply(df$dupe,removeWords,remove.word)
df$dupepercent <- ((str_count(df$dupe, '\\s+')+1)/(str_count(df$x, '\\s+')+1)+(str_count(df$dupe, '\\s+')+1) /(str_count(df$y, '\\s+')+1))/2
df$x <- map2_chr(df$x,dupes,removeWords)
df$y <- map2_chr(df$y,dupes,removeWords)

# 2.5 Remove misc words (words we don't want to match on)
remove.word <- toupper(remove.word)
data[1:2] <- lapply(data[1:2],removeWords,remove.word)
data[1:2] <- lapply(data[1:2],trimws) #minus whitespace

# 2.6 Create Data to remove white spaces (squeeze all words together)
Squeezey <- map_df(data[1:2], function (y) gsub("[[:space:]]", "", y))
Squeezey <- merge(Squeezey$Var1,Squeezey$Var2,by=NULL)
Squeezey <- data.frame(lapply(Squeezey[1:2],as.character),stringsAsFactors = FALSE)

# 2.7 Create data to extract first word of each string for comparison
FW1 <- data.frame(one = word(trimws(data[[1]]),1), stringsAsFactors = FALSE)
FW2 <- data.frame(two = word(trimws(data[[2]]),1), stringsAsFactors = FALSE)
FW <- merge(FW1,FW2,by=NULL)
FW <- data.frame(lapply(FW[1:2],as.character),stringsAsFactors = FALSE)

# 2.8 Remove first word from further comparison
data[1] <- map2_chr(data$Var1,FW1$one,removeWords)
data[2] <- map2_chr(data$Var2,FW2$two,removeWords)
data[1:2] <- lapply(data[1:2],trimws) #minus whitespace

# Section 3 (Calculate Similarity Measures for training) ===================================================================================

# 3.1 Create Dataset for Similarity Scores for "non-first words" 
trainsim <- placeholder
sim <- merge(data$Var1,data$Var2,by=NULL)
sim[1:2] <- data.frame(lapply(sim[1:2],as.character),stringsAsFactors = FALSE)

trainsim$Dlv <- round(stringsim(sim[,1],sim[,2],method = "dl"),3)
trainsim$LCS <- round(stringsim(sim[,1],sim[,2],method = "lcs"),3)
trainsim$QGm <- round(stringsim(sim[,1],sim[,2],method = "qgram"),3)
trainsim$CSn <- round(stringsim(sim[,1],sim[,2],method = "cosine"),3)
trainsim$JCD <- round(stringsim(sim[,1],sim[,2],method = "jaccard"),3)
trainsim$JWd <- round(stringsim(sim[,1],sim[,2],method = "jw"),3)

# 3.2 Convert NA terms of Cosine similarity to 1
trainsim$CSn[is.na(trainsim$CSn)] <- 1 #Cosine similarity measure shows (NA) when strings only have first word

# 3.3 Calculate Similarity Scores for "first word" comparisons


trainsim$FWDlv <- round(stringsim(FW[,1],FW[,2],method = "dl"),3)
trainsim$FWLCS <- round(stringsim(FW[,1],FW[,2],method = "lcs"),3)
trainsim$FWQGm <- round(stringsim(FW[,1],FW[,2],method = "qgram"),3)
trainsim$FWCSn <- round(stringsim(FW[,1],FW[,2],method = "cosine"),3)
trainsim$FWJCD <- round(stringsim(FW[,1],FW[,2],method = "jaccard"),3)
trainsim$FWJWd <- round(stringsim(FW[,1],FW[,2],method = "jw"),3)

# 3.4 Calculate Similarity Scores for "Non-Matching words" comparison

trainsim$NMDlv <- round(stringsim(df$x,df$y,method = "dl"),3)
trainsim$NMLCS <- round(stringsim(df$x,df$y,method = "lcs"),3)
trainsim$NMQGm <- round(stringsim(df$x,df$y,method = "qgram"),3)
trainsim$NMCSn <- round(stringsim(df$x,df$y,method = "cosine"),3)
trainsim$NMJCD <- round(stringsim(df$x,df$y,method = "jaccard"),3)
trainsim$NMJWd <- round(stringsim(df$x,df$y,method = "jw"),3)

# 3.5 Calculate Similarity Scores for squeezed words


trainsim$SqDlv <- round(stringsim(Squeezey$x,Squeezey$y,method = "dl"),3)
trainsim$SqLCS <- round(stringsim(Squeezey$x,Squeezey$y,method = "lcs"),3)
trainsim$SqQGm <- round(stringsim(Squeezey$x,Squeezey$y,method = "qgram"),3)
trainsim$SqCSn <- round(stringsim(Squeezey$x,Squeezey$y,method = "cosine"),3)
trainsim$SqJCD <- round(stringsim(Squeezey$x,Squeezey$y,method = "jaccard"),3)
trainsim$SqJWd <- round(stringsim(Squeezey$x,Squeezey$y,method = "jw"),3)

# 3.6 Attach Dupeword Percentage
trainsim$dupepercent <- df$dupepercent

# Rename first 2 columns
colnames(trainsim)[1:2] <- c("Var1","Var2")

# Set NA values to 0
trainsim[is.na(trainsim)] <- 0


# test TSNE
# tsne <- Rtsne(trainsim[c(-1,-2,-3)],dims=2, verbose = TRUE, check_duplicates = FALSE)
# 
# tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2], col = factor(trainsim$match))
# ggplot(tsne_plot) + geom_point(aes(x=x, y=y, color=col))
# 
# tableau <- cbind(tsne_plot,trainsim[c(1,2,3)])
# write.csv(tableau,"tableau.csv")


# 3.7 Transform data into Principal Components 
PC <- prcomp(trainsim[c(-1,-2,-3)], scale. = T)
TrainPC <- cbind(trainsim[c(1,2,3)],PC$x )


# Section 4 (Train Model)===============================================================================================================

# 4.1 Create train data matrix for xgBoost model
matrix <- xgb.DMatrix(data.matrix(TrainPC[c(-1,-2,-3)]), label = trainsim$match)
matrix2 <- xgb.DMatrix(data.matrix(TrainPC[c(-1,-2,-3)]), label = trainsim$match)

# 4.2 Run Bayesian Optimization to tune hyperparameters of xgBoost model
BO <- invisible(xgb_cv_opt(data = TrainPC[c(-1,-2)],
                  label = match,
                  objectfun = "binary:logistic",
                  evalmetric = "auc",
                  n_folds = 5,
                  acq = "ei",
                  init_points = 10,
                  n_iter = 10,
                  eta_range = c(0.1,1L),
                  max_depth_range = c(4L, 6L),
                  nrounds_range = c(70, 160L),
                  subsample_range = c(0.1, 1L),
                  bytree_range = c(0.4, 1L)
                 ))

#  4.3 Set parameters for tree and linear xgBoost model
params1 <- list(booster = "gbtree", 
                objective = "binary:logistic", 
                eta=BO$Best_Par[[1]], 
                gamma=0, 
                max_depth=BO$Best_Par[[2]], 
                min_child_weight=1, 
                subsample=BO$Best_Par[[4]], 
                colsample_bytree=BO$Best_Par[[5]],
                silent=1
                # ,scale_pos_weight=0.25*count(train$match)[1,2]/count(train$match)[2,2]
)
params2 <- list(booster = "gblinear", 
               objective = "binary:logistic"
               )

# 4.4 Train tree and linear xgBoost models
set.seed(5555)
model1.train <- xgb.train(params = params1, 
                          data = matrix, 
                          nrounds = BO$Best_Par[[3]],
                          watchlist = list( train = matrix),
                          print_every_n = 1,
                          early_stopping_rounds = 10,
                          maximize = TRUE ,
                          eval_metric = "auc")

model2.train <- xgb.train(params = params2, 
                          data = matrix2, 
                          nrounds = BO$Best_Par[[3]],
                          watchlist = list( train = matrix),
                          print_every_n = 1,
                          early_stopping_rounds = 10,
                          maximize = TRUE ,
                          eval_metric = "auc")

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(TrainPC[c(-1,-2,-3)]),model = model1.train)
xgb.plot.importance (importance_matrix = mat) 

mat <- xgb.importance (feature_names = colnames(TrainPC[c(-1,-2,-3)]),model = model2.train)
xgb.plot.importance (importance_matrix = mat) 

# 4.5 Remove unimportant variables and save model data
rm(FW,FW1,FW2,BO,data,df,dupes,mat,matrix,matrix2,params1,params2,placeholder,remove.word,remove.word2,sim,Squeezey,TrainPC,trainsim)
newmodal <- list(model1.train,model2.train,PC)
save(newmodal,file="default_model_data.RData")

# Section 5 (Import test data, clean data, generate datasets for testing )===============================================================================================================

# 5.1 Select words to remove from test dataset (Model will ignore these words when matching)

remove.word2 <- c('FUNDACION,ASOCIACION,CORPORACION,Servicios,Ecuador,Banco,Company,Manufacturing,Services,Europe,Global,National,International,Incorporation,corporation,Company,holdings,holding,limited,Group,corp,gmbh,ltd,llc,inc,pte,pvt,plc,the,
                  bv,nv,lp,l l c,n v,b v,l l p,p l c,SA,S.A.')
remove.word2 <- unlist(strsplit(remove.word2,","))


# 5.2a load Model Data
load("default_model_data.RData")

model1.train <- newmodal[[1]]
model2.train <- newmodal[[2]]
PC <- newmodal[[3]]

# 5.2b Create/Import Test Dataset
Test1 <- read.delim("Table1 - Copy.csv", header=T,sep=",",fill=TRUE, stringsAsFactors = FALSE)
Test2 <- read.delim("Table2 - Copy.csv", header=T,sep=",",fill=TRUE, stringsAsFactors = FALSE)

colnames(Test1)[1] <- "Var1"
colnames(Test2)[1] <- "Var2"


# 5.2c Create Placeholder names
placeholder2 <- merge(Test1[1],Test2[1],by=NULL)

# 5.3a Convert to Uppercase
Test1 <- data.frame(lapply(Test1, toupper),stringsAsFactors = F)
Test2 <- data.frame(lapply(Test2, toupper),stringsAsFactors = F)

# 5.3b Remove non alphanumeric characters
Test1 <- data.frame(lapply(Test1,strip,digit.remove = F, lower.case = F), stringsAsFactors =F)
Test2 <- data.frame(lapply(Test2,strip,digit.remove = F, lower.case = F),stringsAsFactors =F)
Test1 <- map_df(Test1, function (y) gsub("[^A-Za-z0-9]", " ", y))
Test2 <- map_df(Test2, function (y) gsub("[^A-Za-z0-9]", " ", y))

# 5.3c Remove leading and trailing white spaces
Test1 <- data.frame(lapply(Test1, trimws),stringsAsFactors = F)
Test2 <- data.frame(lapply(Test2, trimws),stringsAsFactors = F)

# 5.4a Create Dataset to compare non matching words (Part 1)
df <- merge(Test1[1],Test2[1],by=NULL)

# 5.5 Remove words to not match on
remove.word2 <- toupper(remove.word2)
Test1[1] <- lapply(Test1[1],removeWords,remove.word2)
Test2[1] <- lapply(Test2[1],removeWords,remove.word2)

# 5.6 Create Data to remove white spaces (squeeze all words tgt)
Squeezey1 <- map_df(Test1[1], function (y) gsub("[[:space:]]", "", y))
Squeezey2 <- map_df(Test2[1], function (y) gsub("[[:space:]]", "", y))
Squeezey <- merge(Squeezey1,Squeezey2,by=NULL)
Squeezey <- data.frame(lapply(Squeezey[1:2],as.character),stringsAsFactors = FALSE)
rm(Squeezey1,Squeezey2)

# 5.7 Generate first words data for comparison
FW1 <- data.frame(one = word(trimws(Test1[[1]]),1), stringsAsFactors = FALSE)
FW2 <- data.frame(two = word(trimws(Test2[[1]]),1), stringsAsFactors = FALSE)
FW <- merge(FW1,FW2,by=NULL)
FW <- data.frame(lapply(FW[1:2],as.character),stringsAsFactors = FALSE)


# 5.8 Remove first word from further comparison/ Create dataset of words not including first word
Test1[1] <- map2_chr(Test1$Var1,FW1$one,removeWords)
Test2[1] <- map2_chr(Test2$Var2,FW2$two,removeWords)
Test1[1] <- lapply(Test1[1],trimws)
Test2[1] <- lapply(Test2[1],trimws)
sim <- merge(Test1[1] ,Test2[1] ,all=TRUE)
sim[1:2] <- data.frame(lapply(sim[1:2],as.character),stringsAsFactors = FALSE)
rm(FW1,FW2)

# 5.9a Calculate data to drop (so as to save time)
sim$CSn <- round(stringsim(sim[,1],sim[,2],method = "cosine"),3) 
sim$FWDlv <- round(stringsim(FW[,1],FW[,2],method = "dl"),3)
sim$CSn[is.na(sim$CSn)] <- 2 #Set NAs for Csn to 1 because it is matching to a blank
dropindex <- which((sim$CSn < 0.3 & sim$FWDlv < 0.3)|(sim$CSn = 2 & sim$FWDlv < 0.3))

# 5.9b Pre Filter Data to save time
sim <- sim[-dropindex,]
placeholder2 <- placeholder2[-dropindex,]
FW <- FW[-dropindex,]
Squeezey <- Squeezey[-dropindex,]
df <- df[-dropindex,]


# 5.4b Create data to compare non matching words between strings (Part 2)
colnames(df) <- c("x","y")
df <- data.frame(lapply(df[1:2],as.character),stringsAsFactors = FALSE)
df$dupe <- apply(df,1,function(x)
  paste(Reduce(intersect, strsplit(x, " ")), collapse = " ")) #Matching words
df$dupe[df$dupe==""] <- NA
df$dupesdistinct <- strsplit(df$dupe," ")
df$dupe <- lapply(df$dupe,removeWords,remove.word2)
df$dupepercent <- ((str_count(df$dupe, '\\s+')+1)/(str_count(df$x, '\\s+')+1)+(str_count(df$dupe, '\\s+')+1) /(str_count(df$y, '\\s+')+1))/2
df$x[!is.na(df$dupesdistinct)] <- map2_chr(df$x[!is.na(df$dupesdistinct)],df$dupesdistinct[!is.na(df$dupesdistinct)],removeWords)
df$y[!is.na(df$dupesdistinct)] <- map2_chr(df$y[!is.na(df$dupesdistinct)],df$dupesdistinct[!is.na(df$dupesdistinct)],removeWords)








# Section 6 (Calculate Similarity Measures for testing) =================================================================================
# Create Dataset for Similarity Scores
testsim <- placeholder2

# 6.1 Calculate similarity scores for "non first words'
testsim$Dlv <- round(stringsim(sim[,1],sim[,2],method = "dl"),3)
testsim$LCS <- round(stringsim(sim[,1],sim[,2],method = "lcs"),3)
testsim$QGm <- round(stringsim(sim[,1],sim[,2],method = "qgram"),3)
testsim$CSn <- round(stringsim(sim[,1],sim[,2],method = "cosine"),3)
testsim$JCD <- round(stringsim(sim[,1],sim[,2],method = "jaccard"),3)
testsim$JWd <- round(stringsim(sim[,1],sim[,2],method = "jw"),3)

# Set NA values for cosine similarity to 1
testsim$CSn[is.na(testsim$CSn)] <- 1

# 6.2 Calculate Similarity Scores for "first words"


testsim$FWDlv <- round(stringsim(FW[,1],FW[,2],method = "dl"),3)
testsim$FWLCS <- round(stringsim(FW[,1],FW[,2],method = "lcs"),3)
testsim$FWQGm <- round(stringsim(FW[,1],FW[,2],method = "qgram"),3)
testsim$FWCSn <- round(stringsim(FW[,1],FW[,2],method = "cosine"),3)
testsim$FWJCD <- round(stringsim(FW[,1],FW[,2],method = "jaccard"),3)
testsim$FWJWd <- round(stringsim(FW[,1],FW[,2],method = "jw"),3)

# 6.3 Calculate Similarity Scores for "Non-Matching" words

testsim$NMDlv <- round(stringsim(df$x,df$y,method = "dl"),3)
testsim$NMLCS <- round(stringsim(df$x,df$y,method = "lcs"),3)
testsim$NMQGm <- round(stringsim(df$x,df$y,method = "qgram"),3)
testsim$NMCSn <- round(stringsim(df$x,df$y,method = "cosine"),3)
testsim$NMJCD <- round(stringsim(df$x,df$y,method = "jaccard"),3)
testsim$NMJWd <- round(stringsim(df$x,df$y,method = "jw"),3)

# 6.4 Attach Dupeword Percentage
testsim$dupepercent <- df$dupepercent

# 6.5 Calculate Similarity Scores for squeezed words


testsim$SqDlv <- round(stringsim(Squeezey$Var1,Squeezey$Var2,method = "dl"),3)
testsim$SqLCS <- round(stringsim(Squeezey$Var1,Squeezey$Var2,method = "lcs"),3)
testsim$SqQGm <- round(stringsim(Squeezey$Var1,Squeezey$Var2,method = "qgram"),3)
testsim$SqCSn <- round(stringsim(Squeezey$Var1,Squeezey$Var2,method = "cosine"),3)
testsim$SqJCD <- round(stringsim(Squeezey$Var1,Squeezey$Var2,method = "jaccard"),3)
testsim$SqJWd <- round(stringsim(Squeezey$Var1,Squeezey$Var2,method = "jw"),3)

# Clear unused memory
rm(df,FW,sim,Squeezey,Test1,Test2)
gc()


# Rename first 2 columns
colnames(testsim)[1:2] <- c("Var1","Var2")

# Set NA values to 0
testsim[is.na(testsim)] <- 0

# 6.6 Transform to Principle Components
testPC <- predict(PC,newdata = testsim[c(-1,-2)])

# Section 7 (Calculate Match Test Predictions) ================================================================

# 7.1 transform data into format suitable for model
test <- xgb.DMatrix(as.matrix(testPC))
test2 <- xgb.DMatrix(data.matrix(testPC))

# 7.2 Set Select_Output to be "group" for results to be grouped by input values of the dataset
# Alternative is for results to be shown in decending similarity score value 
Select_Output <- "groupy"

# Top number of matches to display by group (if group is selected)
match_number <- 5

# 7.3 Top number of matches to filter by (if group is not selected)
# (Model calculates a score for every word combination, use this variable to
# filter out most non matches with a lowscore)
top_percentile <- 5

# 7.4 Predict match scores and filter dataset
output <- cbind(placeholder2[1:2],predict(model1.train,newdata=test),predict(model2.train,newdata=test2))
colnames(output)[c(3,4)] <- c("score1","score2")
output$score <- (output$score1+output$score2)/2

output <- output[order(output$score, decreasing = T),]

if (Select_Output == "group")  {
  output <- output %>% 
    arrange(desc(score)) %>%
    group_by(Var2) %>%
    filter(row_number() <= match_number)
  output <- output[order(output$Var2),]
  } else {
  output <- output[output$score > quantile(output$score,prob=1-top_percentile/100),]
}


# 7.5 Write results to output folder
write.csv(output, "testest.csv")

endtime <- Sys.time()
endtime-starttime



Sys.time()-starttime1