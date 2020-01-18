
setwd("C:/Users/AAzqueta/Desktop/Real Estate Project")
getwd()

data_all <- read.table("TOTAL_DATA.csv", header = TRUE, sep=",")

data <- data_all[,1:34]

summary(data)

data <- as.data.frame(data)


## ===================== Descriptive statisitcs Sales ======================== ##

Data_Sale <- data[data$X.operation. == "\"sale\"", ]

## Average price (in millions)

Data_Sale$Price <- Data_Sale$X.price./1000000

par(mfrow=c(1,2))

hist(Data_Sale$Price,
     main="Sale price",
     xlab="Sale Price (in millions)",
     #xlim=c(0,4000),
     col="blue",
     freq=TRUE
)

## Average price
hist(Data_Sale$X.price./Data_Sale$X.size.,
     main="Sale price per square meter",
     xlab="Sale Price (per square meter)",
     #xlim=c(0,4000),
     col="blue",
     freq=TRUE
)


## ===================== Descriptive statisitcs Rent ======================== ##

Data_Rent <- data[data$X.operation. == "\"rent\"", ]

summary(Data_Rent)

par(mfrow=c(1,1))

hist(Data_Rent$X.price.,
     main="Rent price",
     xlab="Rent Price (per month)",
     xlim=c(0,6000),
     col="blue",
     freq=TRUE
)


## ===================== What determines the Sale price ======================== ##
# https://uc-r.github.io/random_forests
# Random forest

library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform

data_Sale_Analysis <- Data_Sale[ ,c(7,9,12,15,20,21,23,24,26,30,31)]

data_Sale_Analysis <- transform(
  data_Sale_Analysis,
  X.price.=as.integer(X.price.),
  X.priceByArea.=as.integer(X.priceByArea.),
  #X.district. =as.factor(X.district.),
  X.floor. =as.factor(X.floor.),
  X.exterior. =as.factor(X.exterior.),
  X.hasLift. =as.factor(X.hasLift.),
  X.isParkingSpaceIncludedInPrice. =as.factor(X.isParkingSpaceIncludedInPrice.),
  #X.municipality. =as.factor(X.municipality.),
  X.newDevelopment. =as.factor(X.newDevelopment.),
  X.numPhotos. =as.integer(X.numPhotos.),
  X.propertyType. =as.factor(X.propertyType.),
  X.size. =as.integer(X.size.),
  X.status. =as.factor(X.status.)
  
)


set.seed(123)
ames_split <- initial_split(data_Sale_Analysis, prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

set.seed(123)

# default RF model
m1 <- randomForest(
  formula = X.price. ~ .,
  data    = ames_train
)

m1

plot(m1)

# number of trees with lowest MSE
which.min(m1$mse)

# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])


# create training and validation data 
set.seed(123)
valid_split <- initial_split(ames_train, .8)

# training data
ames_train_v2 <- analysis(valid_split)


# validation data
ames_valid <- assessment(valid_split)
x_test <- ames_valid[setdiff(names(ames_valid), "X.price.")]
y_test <- ames_valid$X.price.


rf_oob_comp <- randomForest(
  formula = X.price. ~ .,
  data    = ames_train_v2,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous(labels = scales::dollar) +
  xlab("Number of trees")


# names of features
features <- setdiff(names(ames_train), "X.price.")

set.seed(123)

m2 <- tuneRF(
  x          = ames_train[features],
  y          = ames_train$X.price.,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)


# randomForest speed
system.time(
  ames_randomForest <- randomForest(
    formula = X.price. ~ ., 
    data    = ames_train, 
    ntree   = 500,
    mtry    = floor(length(features) / 3)
  )
)
##    user  system elapsed 
##  55.371   0.590  57.364

# ranger speed
system.time(
  ames_ranger <- ranger(
    formula   = X.price. ~ ., 
    data      = ames_train, 
    num.trees = 500,
    mtry      = floor(length(features) / 3)
  )
)


# hyperparameter grid search
hyper_grid <- expand.grid(
  mtry       = seq(20, 30, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

# total number of combinations
nrow(hyper_grid)

for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)



OOB_RMSE <- vector(mode = "numeric", length = 10)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = X.price. ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = 5,
    min.node.size   = 2,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}



hist(OOB_RMSE, breaks = 2)


optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(5) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 5 important variables")






