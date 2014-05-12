##### Preparing Training Data:
# Load the data:
train1 <- read.csv("training.csv", head = TRUE, sep=",", quote="\"", na.strings="\\N")
state_names <- c("AR","AZ","CA","CO","CT","DC","DE","FL","GA","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME",
                 "MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
                 "TN","TX","UT","VA","VT","WA","WI","WV","WY")

# Test the proportion of data is labeled as 1
length(which(train1$booking_ind == 1))/nrow(train1)

# Fund_date was recorded after labeled 1 in target variable
# Thus, it doesn't have the predictive functionality.
# Remove this feature.
train1 <- train1[, !(names(train1) == "fund_date")]

# Data Cleaning.
for (n in names(train1)) { 
  col <- train1[,n] # each feature
  na <- is.na( col ) # check whether feature contains NA
  # check whether majority (99.9%)  of feature values are NA. If so, then remove the feature.
  if ((length(na)-sum(na))/length(na)<0.001) {
    print(paste("X",n))
    train1 <- train1[,!(names(train1) == n)]
    next
  }
  if (any(na)) {
    if (is.numeric(col)) {
      print(paste("*",n))
      train1[, paste(n, "NA", sep = "_")] <- na
      train1[na,n] <- median(col, na.rm=T) # If NA is in the numeric feature, fit the NA with median values.
    } else {
      print(paste(".",n))
      train1[na, n] <- '#NA#' # If NA is in the categorical feature, then create a new category as '#NA#'.
    }
  } 
  # Add more features to the dataset.
  dt <- strptime(col, "%m/%d/%Y")
  if(!any(is.na(dt))) { 
    print(paste("!",n))
    train1 <- train1[,!(names(train1) == n)]
    train1[,paste(n, "wday", sep = "_")] <- dt$wday   # the day in a week (start from Monday)
    train1[,paste(n, "mday", sep = "_")] <- dt$mday   # the day in a month
    train1[,paste(n, "yday", sep = "_")] <- dt$yday   # the day in a year
    train1[,paste(n, "year", sep = "_")] <- dt$year+1900   # the year in 21st century
  }
}

# Create dummy variables of term, car_type, partner_cat, and cust_seg.
term <- as.data.frame(model.matrix(~factor(term), data=train1)[,-1])
colnames(term) <- c("term_48", "term_60", "term_66", "term_72")
train1 <- cbind(train1, term)
car_type <- as.data.frame(model.matrix(~factor(car_type), data=train1)[,-1])
colnames(car_type) <- c("car_type_U")
train1 <- cbind(train1, car_type)
partner_cat <- model.matrix(~factor(partner_cat), data=train1)[,-1]
colnames(partner_cat) <- c("partner_cat_2", "partner_cat_3")
train1 <- cbind(train1, partner_cat)
cust_seg <- model.matrix(~factor(cust_seg), data=train1)[,-1]
colnames(cust_seg) <- c("cust_seg_B", "cust_seg_C", "cust_seg_D")
train1 <- cbind(train1, cust_seg)

# Deal with the customer without State.
# Convert state to a dummy variable.
userWithoutState <- train1[which(train1$state == ""), ]
train1 <- train1[-which(train1$state == ""), ]  # Drop the user without state will be add back later
state <- model.matrix(~factor(state), data=train1)[,-1]
colnames(state) <- state_names
train1 <- cbind(train1, state)
train1$state <- factor(train1$state)
aa <- as.data.frame(matrix(0, 1, length(state_names)))
colnames(aa) <- state_names
# Add back the user without state.
userWithoutState <- cbind(userWithoutState, aa)
train1 <- rbind(train1, userWithoutState)

# Remove original features term, car_type, partner_cat, cust_seg and state.
train1 <- train1[,!(names(train1) %in% c("term", "car_type", "partner_cat", "cust_seg", "state"))]


########## Preparing Test Dataset:
# Load the data:
test1 <- read.csv("testing.csv", head = TRUE, sep=",", quote="\"", na.strings="\\N")

# Test the proportion of data is labeled as 1
length(which(test1$booking_ind == 1))/nrow(test1)

# Remove feature fund_date
test1 <- test1[, !(names(test1) == "fund_date")]

# Data Cleaning.
for (n in names(test1)) { 
  col <- test1[,n] # each feature
  na <- is.na( col ) # check whether feature contains NA
  # check whether majority (99.9%)  of feature values are NA. If so, then remove the feature.
  if ((length(na)-sum(na))/length(na)<0.001) {
    print(paste("X",n))
    test1 <- test1[,!(names(test1) == n)]
    next
  }
  if (any(na)) {
    if (is.numeric(col)) {
      print(paste("*",n))
      test1[, paste(n, "NA", sep = "_")] <- na
      test1[na,n] <- median(col, na.rm=T) # If NA is in the numeric feature, fit the NA with median values.
    } else {
      print(paste(".",n))
      test1[na, n] <- '#NA#' # If NA is in the categorical feature, then create a new category as '#NA#'.
    }
  } 
  # Add more features to the dataset.
  dt <- strptime(col, "%m/%d/%Y")
  if(!any(is.na(dt))) { 
    print(paste("!",n))
    test1 = test1[,!(names(test1) == n)]
    test1[,paste(n, "wday", sep = "_")] <- dt$wday   # the day in a week (start from Monday)
    test1[,paste(n, "mday", sep = "_")] <- dt$mday   # the day in a month
    test1[,paste(n, "yday", sep = "_")] <- dt$yday   # the day in a year
    test1[,paste(n, "year", sep = "_")] <- dt$year+1900  # the year in 21st century
  }
}

# Create dummy variables of term, car_type, partner_cat, cust_seg and state.
term <- as.data.frame(model.matrix(~factor(term), data=test1)[,-1])
colnames(term) <- c("term_48", "term_60", "term_66", "term_72")
test1 <- cbind(test1, term)
car_type <- as.data.frame(model.matrix(~factor(car_type), data=test1)[,-1])
colnames(car_type) <- c("car_type_U")
test1 <- cbind(test1, car_type)
partner_cat <- model.matrix(~factor(partner_cat), data=test1)[,-1]
colnames(partner_cat) <- c("partner_cat_2", "partner_cat_3")
test1 <- cbind(test1, partner_cat)
cust_seg <- model.matrix(~factor(cust_seg), data=test1)[,-1]
colnames(cust_seg) <- c("cust_seg_B", "cust_seg_C", "cust_seg_D")
test1 <- cbind(test1, cust_seg)
state <- model.matrix(~factor(state), data=test1)[,-1]
colnames(state) <- state_names
test1 <- cbind(test1, state)

# Remove original features car_type, partner_cat, cust_seg and state
test1 <- test1[,!(names(test1) %in% c("term", "car_type", "partner_cat", "cust_seg", "state"))]


########## Preparing Price Optimization Dataset:
# Load the data:
opt1 <- read.csv("other.csv", head = TRUE, sep=",", quote="\"", na.strings="\\N")

# Test the proportion of data is labeled as 1
length(which(opt1$booking_ind == 1))/nrow(opt1)

# Remove feature fund_date
opt1 <- opt1[, !(names(opt1) == "fund_date")]

# Data Cleaning.
for (n in names(opt1)) { 
  col <- opt1[,n] # each feature
  na <- is.na( col ) # check whether feature contains NA
  # check whether majority (99.9%)  of feature values are NA. If so, then remove the feature.
  if ((length(na)-sum(na))/length(na)<0.001) {
    print(paste("X",n))
    opt1 <- opt1[,!(names(opt1) == n)]
    next
  }
  if (any(na)) {
    if (is.numeric(col)) {
      print(paste("*",n))
      opt1[, paste(n, "NA", sep = "_")] <- na
      opt1[na,n] <- median(col, na.rm=T) # If NA is in the numeric feature, fit the NA with median values.
    } else {
      print(paste(".",n))
      opt1[na, n] <- '#NA#' # If NA is in the categorical feature, then create a new category as '#NA#'.
    }
  } 
  # Add more features to the dataset.
  dt <- strptime(col, "%m/%d/%Y")
  if(!any(is.na(dt))) { 
    print(paste("!",n))
    opt1 = opt1[,!(names(opt1) == n)]
    opt1[,paste(n, "wday", sep = "_")] <- dt$wday   # the day in a week (start from Monday)
    opt1[,paste(n, "mday", sep = "_")] <- dt$mday   # the day in a month
    opt1[,paste(n, "yday", sep = "_")] <- dt$yday   # the day in a year
    opt1[,paste(n, "year", sep = "_")] <- dt$year+1900   # the year in 21st century
  }
}

# Create dummy variables of car_type, partner_cat and cust_seg.
car_type <- as.data.frame(model.matrix(~factor(car_type), data=opt1)[,-1])
colnames(car_type) <- c("car_type_U")
opt1 <- cbind(opt1, car_type)
partner_cat <- model.matrix(~factor(partner_cat), data=opt1)[,-1]
colnames(partner_cat) <- c("partner_cat_2", "partner_cat_3")
opt1 <- cbind(opt1, partner_cat)
cust_seg <- model.matrix(~factor(cust_seg), data=opt1)[,-1]
colnames(cust_seg) <- c("cust_seg_B", "cust_seg_C", "cust_seg_D")
opt1 <- cbind(opt1, cust_seg)

# Create dummy variables of state.
for (state in state_names){
  opt1[, state] <- rep(0, nrow(opt1))
}
for (i in 1:nrow(opt1)){
  if (opt1$state[i] %in% state_names) {
    index <- which(state_names == opt1$state[i])
    opt1[i, state_names[index]] <- 1
  }
}

# Create dummy variables of term.
term_dummy <- c("term_48", "term_60", "term_66", "term_72")
for (term in term_dummy){
  opt1[, term] <- rep(0, nrow(opt1))
}
for (i in 1:nrow(opt1)){
  if (opt1$term[i] %in% term_dummy) {
    index <- which(term_dummy == opt1$term[i])
    opt1[i, term_dummy[index]] <- 1
  }
}

# Remove original features term, car_type, partner_cat, cust_seg and state.
opt1 <- opt1[,!(names(opt1) %in% c("term", "car_type", "partner_cat", "cust_seg", "state"))]

###### Write output to csv files
write.csv(train1,"train2.csv", row.names=F)
write.csv(test1,"test2.csv", row.names=F)
write.csv(opt1,"other2.csv", row.names=F)

