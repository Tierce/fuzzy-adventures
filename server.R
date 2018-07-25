#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#
library(shiny)

options(shiny.maxRequestSize=30*1024^2)
shinyServer(function(input, output,session) {

  # Import train dataset
  data1 <- reactive({
    
    inFile1 <- input$file1
    
    if (is.null(inFile1))
      return(NULL)
    
    tbl <- read.delim(inFile1$datapath, header=T,sep=",", stringsAsFactors = F)
    tbl
  })

  # Import reference dataset
  TestA <- reactive({

    inFile2 <- input$file2

    if (is.null(inFile2))
      return(NULL)

    tbl <- read.delim(inFile2$datapath, header=T,sep=",", stringsAsFactors = F)
    tbl
  })

  # Import test dataset
  TestB <- reactive({

    inFile3 <- input$file3

    if (is.null(inFile3))
      return(NULL)

    tbl <- read.delim(inFile3$datapath, header=T,sep=",", stringsAsFactors = F)
    tbl
  })
  
  # Imput words to ignore 
  RMwords2 <- reactive({
    
    inFile3 <- input$RMwords2
    inFile3 <- unlist(strsplit(inFile3,","))
    
    
    if (is.null(inFile3))
      return(NULL)
    inFile3
  })

  modeldata <- eventReactive(input$go2,{
    # Section 2 (Upload train data, clean data, generate datasets for training) =============================================
    
    # Disable button 
    updateButton(session,'go2', disabled = TRUE)
    
    # Create a Progress object (To display a progress bar)
    progress <- shiny::Progress$new()
    # Make sure it closes when the code function finishes, even if there's an error
    on.exit(progress$close())
    
    #Increase progress bar 1
    progress$set(message="Model running")
    
    
    #Set words to remove from train data
    remove.word <- c('International', 'Incorporation',"corporation",'Company',"holdings","holding","limited",'Group','corp',"gmbh","ltd","llc","inc","pte","pvt","plc","the",
                     "bv","nv","lp","l l c","n v","b v","l l p","p l c", 'SA')
    
    # Import and Clean data
    data <- data1()
    N <- nrow(data)
    data[1:2] <- lapply(data[1:2], toupper) #convert to uppercase
    data[1:2] <- lapply(data[1:2],strip,digit.remove = F, lower.case = F) #minus punctuation
    data[1:2] <- map_df(data[1:2], function (y) gsub("[^A-Za-z0-9]", " ", y)) #Minus non alphanumeric characters
    data[1:2] <- lapply(data[1:2],trimws) #minus whitespace
    
    #Increase progress bar 2
    progress$set(1/7, detail = paste("Importing train data"))
    
    #Label Matches
    colnames(data) <- c("Var1","Var2")
    data$match <- 1
    placeholder <- merge(data$Var1,data$Var2,by=NULL)
    colnames(placeholder) <- c("Var1","Var2")
    placeholder<- join(placeholder,data,by=c("Var1","Var2"))
    placeholder$match[is.na(placeholder$match)] <- 0
    
    #Increase progress bar 3
    progress$set(2/7, detail = paste("Transforming data"))
    
    #Create data to compare non-matching words within strings
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
    
    # Remove misc words (words we don't want to match on)
    remove.word <- toupper(remove.word)
    data[1:2] <- lapply(data[1:2],removeWords,remove.word)
    data[1:2] <- lapply(data[1:2],trimws) #minus whitespace
    
    #Create Data to remove white spaces (squeeze all words together)
    Squeezey <- map_df(data[1:2], function (y) gsub("[[:space:]]", "", y))
    
    #Create data to extract first word of each string for comparison
    FW1 <- data.frame(one = word(trimws(data[[1]]),1), stringsAsFactors = FALSE)
    FW2 <- data.frame(two = word(trimws(data[[2]]),1), stringsAsFactors = FALSE)
    
    #Remove first word from further comparison
    data[1] <- map2_chr(data$Var1,FW1$one,removeWords)
    data[2] <- map2_chr(data$Var2,FW2$two,removeWords)
    data[1:2] <- lapply(data[1:2],trimws) #minus whitespace
    
    # Section 3 (Calculate Similarity Measures for training) ===================================================================================
    
    #Increase progress bar 4
    progress$set(3/7, detail = paste("Generating Similarity Measures"))
    
    # Create Dataset for Similarity Scores for "non-first words"
    trainsim <- placeholder
    sim <- merge(data$Var1,data$Var2,by=NULL)
    sim[1:2] <- data.frame(lapply(sim[1:2],as.character),stringsAsFactors = FALSE)
    
    trainsim$Dlv <- round(stringsim(sim[,1],sim[,2],method = "dl"),3)
    trainsim$LCS <- round(stringsim(sim[,1],sim[,2],method = "lcs"),3)
    trainsim$QGm <- round(stringsim(sim[,1],sim[,2],method = "qgram"),3)
    trainsim$CSn <- round(stringsim(sim[,1],sim[,2],method = "cosine"),3)
    trainsim$JCD <- round(stringsim(sim[,1],sim[,2],method = "jaccard"),3)
    trainsim$JWd <- round(stringsim(sim[,1],sim[,2],method = "jw"),3)
    
    # Convert NA terms of Cosine similarity to 1
    trainsim$CSn[is.na(trainsim$CSn)] <- 1 #Cosine similarity measure shows (NA) when strings only have first word
    
    #Calculate Similarity Scores for "first word" comparisons
    FW <- merge(FW1,FW2,by=NULL)
    FW <- data.frame(lapply(FW[1:2],as.character),stringsAsFactors = FALSE)
    
    trainsim$FWDlv <- round(stringsim(FW[,1],FW[,2],method = "dl"),3)
    trainsim$FWLCS <- round(stringsim(FW[,1],FW[,2],method = "lcs"),3)
    trainsim$FWQGm <- round(stringsim(FW[,1],FW[,2],method = "qgram"),3)
    trainsim$FWCSn <- round(stringsim(FW[,1],FW[,2],method = "cosine"),3)
    trainsim$FWJCD <- round(stringsim(FW[,1],FW[,2],method = "jaccard"),3)
    trainsim$FWJWd <- round(stringsim(FW[,1],FW[,2],method = "jw"),3)
    
    # Calculate Similarity Scores for "Non-Matching words" comparison
    
    trainsim$NMDlv <- round(stringsim(df$x,df$y,method = "dl"),3)
    trainsim$NMLCS <- round(stringsim(df$x,df$y,method = "lcs"),3)
    trainsim$NMQGm <- round(stringsim(df$x,df$y,method = "qgram"),3)
    trainsim$NMCSn <- round(stringsim(df$x,df$y,method = "cosine"),3)
    trainsim$NMJCD <- round(stringsim(df$x,df$y,method = "jaccard"),3)
    trainsim$NMJWd <- round(stringsim(df$x,df$y,method = "jw"),3)
    
    # Calculate Similarity Scores for squeezed words
    Squeezey <- merge(Squeezey$Var1,Squeezey$Var2,by=NULL)
    Squeezey <- data.frame(lapply(Squeezey[1:2],as.character),stringsAsFactors = FALSE)
    
    trainsim$SqDlv <- round(stringsim(Squeezey$x,Squeezey$y,method = "dl"),3)
    trainsim$SqLCS <- round(stringsim(Squeezey$x,Squeezey$y,method = "lcs"),3)
    trainsim$SqQGm <- round(stringsim(Squeezey$x,Squeezey$y,method = "qgram"),3)
    trainsim$SqCSn <- round(stringsim(Squeezey$x,Squeezey$y,method = "cosine"),3)
    trainsim$SqJCD <- round(stringsim(Squeezey$x,Squeezey$y,method = "jaccard"),3)
    trainsim$SqJWd <- round(stringsim(Squeezey$x,Squeezey$y,method = "jw"),3)
    
    # Attach Dupeword Percentage
    trainsim$dupepercent <- df$dupepercent
    
    # Rename first 2 columns
    colnames(trainsim)[1:2] <- c("Var1","Var2")
    
    # Set NA values to 0
    trainsim[is.na(trainsim)] <- 0
    
    # Transform data into Principal Components
    PC <- prcomp(trainsim[c(-1,-2,-3)], scale. = T)
    TrainPC <- cbind(trainsim[c(1,2,3)],PC$x )
    
    
    # Section 4 (Train Model)===============================================================================================================
    
    #Increase progress bar 5
    progress$set(0.5, detail = paste("Conducting Bayesian Optimization of model hyperparameters"))
    
    
    # Create train data matrix for xgBoost model
    matrix <- xgb.DMatrix(data.matrix(TrainPC[c(-1,-2,-3)]), label = trainsim$match)
    matrix2 <- xgb.DMatrix(data.matrix(TrainPC[c(-1,-2,-3)]), label = trainsim$match)
    
    # Run Bayesian Optimization to tune hyperparameters of xgBoost model
    
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
    
    
    #Increase progress bar 6
    progress$set(6/7, detail = paste("Training xgBoost model"))
    
    # Set parameters for tree and linear xgBoost model
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
    
    # Train tree and linear xgBoost models
    set.seed(5555)
    model1.train <- xgb.train(params = params1,
                              data = matrix,
                              nrounds = BO$Best_Par[[3]],
                              watchlist = list( train = matrix),
                              print_every_n = 1,
                              early_stopping_rounds = 10,
                              maximize = TRUE ,
                              eval_metric = "auc",
                              verbose = 0)
    
    model2.train <- xgb.train(params = params2,
                              data = matrix2,
                              nrounds = BO$Best_Par[[3]],
                              watchlist = list( train = matrix),
                              print_every_n = 1,
                              early_stopping_rounds = 10,
                              maximize = TRUE ,
                              eval_metric = "auc",
                              verbose=0)
    
    #Increase progress bar 7
    progress$set(1, detail = paste("Saving Model"))
    
    # #view variable importance plot
    # mat <- xgb.importance (feature_names = colnames(TrainPC[c(-1,-2,-3)]),model = model1.train)
    # xgb.plot.importance (importance_matrix = mat)
    # 
    # mat <- xgb.importance (feature_names = colnames(TrainPC[c(-1,-2,-3)]),model = model2.train)
    # xgb.plot.importance (importance_matrix = mat)
    
    #Reenable button
    updateButton(session,'go2', disabled = TRUE)
    
    # Remove unimportant variables
    rm(FW,FW1,FW2,BO,data,df,dupes,matrix,matrix2,params1,params2,placeholder,remove.word,sim,Squeezey,TrainPC,trainsim)
    
    list(model1.train,model2.train,PC)
    
  })
  observe({modeldata()})
    
  matchdata <- eventReactive(input$go,{
    # Disable button
    updateButton(session,"go", disabled = TRUE)
    updateButton(session,"help", disabled = TRUE)
    
    # Create a Progress object (To display a progress bar)
    progress <- shiny::Progress$new()
    # Make sure it closes when the code function finishes, even if there's an error
    on.exit(progress$close())
    

    

    # Section 5 (Import test data, clean data, generate datasets for testing )===============================================================================================================
    
    # Select words to remove from test dataset (Model will ignore these words when matching)
    remove.word2 <- RMwords2()
    
    #Open progress bar
    progress$set(message = "Model running")
    
    # Load models
    
    if(input$modelchoose=='upload'){
      inFile4 <- input$file4
      
      if (is.null(inFile4))
        return(NULL)
      
      
      load(inFile4$datapath)
    }else{
    load("default_model_data.RData")}
    
    model1.train <- newmodal[[1]]
    model2.train <- newmodal[[2]]
    PC <- newmodal[[3]]
    
    
    
    #Increase progress bar 1
    progress$set(1/6, detail = paste("Preparing test data"))
    
    
    #Create/Import Test Dataset
    Test1 <- TestA()
    Test2 <- TestB()
    
    colnames(Test1)[1] <- "Var1"
    colnames(Test2)[1] <- "Var2"

    
    #Create Placeholder names
    placeholder2 <- merge(Test1[1],Test2[1],by=NULL)
    
    #Convert to Uppercase
    Test1 <- data.frame(lapply(Test1, toupper),stringsAsFactors = F)
    Test2 <- data.frame(lapply(Test2, toupper),stringsAsFactors = F)
    
    #Remove non alphanumeric characters
    Test1 <- data.frame(lapply(Test1,strip,digit.remove = F, lower.case = F), stringsAsFactors =F)
    Test2 <- data.frame(lapply(Test2,strip,digit.remove = F, lower.case = F),stringsAsFactors =F)
    Test1 <- map_df(Test1, function (y) gsub("[^A-Za-z0-9]", " ", y))
    Test2 <- map_df(Test2, function (y) gsub("[^A-Za-z0-9]", " ", y))
    
    #Remove leading and trailing white spaces
    Test1 <- data.frame(lapply(Test1, trimws),stringsAsFactors = F)
    Test2 <- data.frame(lapply(Test2, trimws),stringsAsFactors = F)
    
    #Increase progress bar 2
    progress$set(2/6, detail = paste("Transforming test data (This takes a long time)"))
    
    # Create Dataset to compare non matching words (Part 1)
    df <- merge(Test1[1],Test2[1],by=NULL)
    
    #Remove words to not match on
    remove.word2 <- toupper(remove.word2)
    Test1[1] <- lapply(Test1[1],removeWords,remove.word2)
    Test2[1] <- lapply(Test2[1],removeWords,remove.word2)
    
    #Create Data to remove white spaces (squeeze all words tgt)
    Squeezey1 <- map_df(Test1[1], function (y) gsub("[[:space:]]", "", y))
    Squeezey2 <- map_df(Test2[1], function (y) gsub("[[:space:]]", "", y))
    Squeezey <- merge(Squeezey1,Squeezey2,by=NULL)
    Squeezey <- data.frame(lapply(Squeezey[1:2],as.character),stringsAsFactors = FALSE)
    rm(Squeezey1,Squeezey2)
    
    #Generate first words data for comparison
    FW1 <- data.frame(one = word(trimws(Test1[[1]]),1), stringsAsFactors = FALSE)
    FW2 <- data.frame(two = word(trimws(Test2[[1]]),1), stringsAsFactors = FALSE)
    FW <- merge(FW1,FW2,by=NULL)
    FW <- data.frame(lapply(FW[1:2],as.character),stringsAsFactors = FALSE)
    
    
    #Remove first word from further comparison/ Create dataset of words not including first word
    Test1[1] <- map2_chr(Test1$Var1,FW1$one,removeWords)
    Test2[1] <- map2_chr(Test2$Var2,FW2$two,removeWords)
    Test1[1] <- lapply(Test1[1],trimws)
    Test2[1] <- lapply(Test2[1],trimws)
    sim <- merge(Test1[1] ,Test2[1] ,all=TRUE)
    sim[1:2] <- data.frame(lapply(sim[1:2],as.character),stringsAsFactors = FALSE)
    rm(FW1,FW2)
    
    # Calculate data to drop (so as to save time)
    sim$CSn <- round(stringsim(sim[,1],sim[,2],method = "cosine"),3) 
    sim$FWDlv <- round(stringsim(FW[,1],FW[,2],method = "dl"),3)
    sim$CSn[is.na(sim$CSn)] <- 2 #Set NAs for Csn to 1 because it is matching to a blank
    dropindex <- which((sim$CSn < 0.3 & sim$FWDlv < 0.3)|(sim$CSn = 2 & sim$FWDlv < 0.3))
    
    # Pre Filter Data to save time
    sim <- sim[-dropindex,]
    placeholder2 <- placeholder2[-dropindex,]
    FW <- FW[-dropindex,]
    Squeezey <- Squeezey[-dropindex,]
    df <- df[-dropindex,]
    
    
    # Create data to compare non matching words between strings (Part 2)
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
    
    #Increase progress bar 3
    progress$set(5/6, detail = paste("Generating Similarity Measures"))
    
    # Create Dataset for Similarity Scores
    testsim <- placeholder2
    
    # Calculate similarity scores for "non first words'
    testsim$Dlv <- round(stringsim(sim[,1],sim[,2],method = "dl"),3)
    testsim$LCS <- round(stringsim(sim[,1],sim[,2],method = "lcs"),3)
    testsim$QGm <- round(stringsim(sim[,1],sim[,2],method = "qgram"),3)
    testsim$CSn <- round(stringsim(sim[,1],sim[,2],method = "cosine"),3)
    testsim$JCD <- round(stringsim(sim[,1],sim[,2],method = "jaccard"),3)
    testsim$JWd <- round(stringsim(sim[,1],sim[,2],method = "jw"),3)
    
    # Set NA values for cosine similarity to 1
    testsim$CSn[is.na(testsim$CSn)] <- 1
    
    #Calculate Similarity Scores for "first words"
    
    testsim$FWDlv <- round(stringsim(FW[,1],FW[,2],method = "dl"),3)
    testsim$FWLCS <- round(stringsim(FW[,1],FW[,2],method = "lcs"),3)
    testsim$FWQGm <- round(stringsim(FW[,1],FW[,2],method = "qgram"),3)
    testsim$FWCSn <- round(stringsim(FW[,1],FW[,2],method = "cosine"),3)
    testsim$FWJCD <- round(stringsim(FW[,1],FW[,2],method = "jaccard"),3)
    testsim$FWJWd <- round(stringsim(FW[,1],FW[,2],method = "jw"),3)
    
    # Calculate Similarity Scores for "Non-Matching" words
    
    testsim$NMDlv <- round(stringsim(df$x,df$y,method = "dl"),3)
    testsim$NMLCS <- round(stringsim(df$x,df$y,method = "lcs"),3)
    testsim$NMQGm <- round(stringsim(df$x,df$y,method = "qgram"),3)
    testsim$NMCSn <- round(stringsim(df$x,df$y,method = "cosine"),3)
    testsim$NMJCD <- round(stringsim(df$x,df$y,method = "jaccard"),3)
    testsim$NMJWd <- round(stringsim(df$x,df$y,method = "jw"),3)
    
    # Attach Dupeword Percentage
    testsim$dupepercent <- df$dupepercent
    
    # Calculate Similarity Scores for squeezed words
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
    
    #Transform to Principle Components
    testPC <- predict(PC,newdata = testsim[c(-1,-2)])
    
    # Section 7 (Calculate Match Test Predictions) ================================================================
    
    #Increase progress bar 4
    progress$set(1, detail = paste("Generating Predictions"))
    
    test <- xgb.DMatrix(as.matrix(testPC))
    test2 <- xgb.DMatrix(data.matrix(testPC))
    
    # Predict match scores
    output <- cbind(placeholder2[1:2],predict(model1.train,newdata=test),predict(model2.train,newdata=test2))
    colnames(output)[c(1,2,3,4)] <- c("Reference","Check", "score1","score2")
    output$score <- (output$score1+output$score2)/2
    output[c(3,4,5)] <- round(output[c(3,4,5)],3)
    
    # Enable Button
    updateButton(session,"go", disabled = FALSE)
    updateButton(session,"help", disabled = FALSE)
    
    output <- output[order(output$score, decreasing = T),]
    output
    
  })
  

  #Create object to hold new generated model before download
  newmodal <- reactiveValues()
  observe({
    if(!is.null(modeldata()))
      isolate(
        newmodal <<- modeldata()
      )
  })
  
  
  # Group Matches?
  match <- reactive({
    output <- matchdata()
   
    # Top number of matches to filter by (if group is not selected)
    # (Model calculates a score for every word combination, use this variable to
    # filter out most non matches with a lowscore)
    top_percentile <- input$percentile
    

    if (input$group)  {
      output <- output %>% 
        arrange(desc(score)) %>%
        group_by(Check) %>%
        filter(row_number() <= input$groupselect)
      output <- output[order(output$Check),]
    } else {
      output <- output[output$score > quantile(output$score,prob=1-top_percentile/100),]
    }
    output
    
  })
  
  # Display matches in UI
  output$displaytable <- renderDataTable({datatable(match())})
  
  # Downloadable csv of match dataset ----
  output$matched.csv <- downloadHandler(
    filename = function() {"match_data.csv"},
    content = function(file) {write.csv(match(), file)},
    contentType = "text/csv"
  )
  
  # Downloadable workspace of new model  ----
  output$newmodel <- downloadHandler(
    filename = function() {paste("newmodel",Sys.time(),".Rdata", sep = "") },
    content = function(file) {save(newmodal, file = file)}
  )
  
  #Code for help button
  # start introjs when button is pressed with custom options and events
  observeEvent(input$help,
               introjs(session, options = list("nextLabel"="Next",
                                               "prevLabel"="Back",
                                               "skipLabel"="Close")
                      )
  )
  

  
  # Stop R from running after browser is closed
  session$onSessionEnded(stopApp)
})
