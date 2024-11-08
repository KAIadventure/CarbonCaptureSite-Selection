library (data.table)
library (car)
library(caTools)
library (corrplot)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)


setwd("your working directory")
ccs.dt <- fread("co2storage.csv", stringsAsFactors = T)
attach (ccs.dt)

dim(ccs.dt)
summary (ccs.dt)

#--------------------------------data exploration and visulisation------------------------------------------------------
co2_capacity_totals.Basin <- aggregate(CO2.storage.capacity ~ Basin, data = ccs.dt, FUN = sum)
co2_capacity_totals.Basin
#The top 3 Basin with the highest co2capacity are EAB, RAKB, IHCA

co2_capacity_totals.Formation <- aggregate(CO2.storage.capacity ~ Formation, data = ccs.dt, FUN = sum)
co2_capacity_totals.Formation
# wasia/mishrif have the highest co2 capacity 

co2_capacity_totals.On.off.shore <- aggregate(CO2.storage.capacity ~ On.off.shore, data = ccs.dt, FUN = sum)
co2_capacity_totals.On.off.shore
# onshore basin have greater capacity 

Basin.Area <- aggregate(Area ~ Basin, data = ccs.dt, FUN = sum)
Basin.Area
# EAB have the largest area

co2_capacity_totals <- sum(CO2.storage.capacity)
co2_capacity_totals
# total co2 capacity of Saudi Arabia approximately 460Gt

co2_capacity_per_area <- co2_capacity_totals.Basin / Basin.Area
co2_capacity_per_area
# Ummluj have the highest capacity per area

co2_capacity_per_area_table <- data.frame(co2_capacity_totals.Basin,
  "Area" = Basin.Area$Area,
  "Capacity per unit Area" = co2_capacity_per_area$CO2.storage.capacity)
co2_capacity_per_area_table

par(mfrow = c(1,2))

barplot(co2_capacity_totals.Basin$CO2.storage.capacity / co2_capacity_totals, 
        names.arg = co2_capacity_totals.Basin$Basin, 
        xlab = "Basin", ylab = "Proportion of CO2 Capacity", col = "skyblue", 
        main = "Proportion of CO2 Capacity by Basin")
barplot(Basin.Area$Area, names.arg = Basin.Area$Basin, xlab = "Basin", ylab = "Area", col = "skyblue", main = "Area of Each Basin")
# As expected, co2 capacity is proportion to the area of the basin
barplot(co2_capacity_per_area_table$Capacity.per.unit.Area, 
        names.arg = Basin.Area$Basin, 
        xlab = "Basin", 
        ylab = "Capacity per unit Area", 
        col = "skyblue", 
        main = "Capacity per unit Area of Each Basin")


par(mfrow = c(1,1))

# Calculate the proportions
proportions <- co2_capacity_totals.Basin$CO2.storage.capacity / co2_capacity_totals

# Create a pie chart without labels
pie(proportions, main = "Proportion of CO2 Capacity by Basin", col = rainbow(length(proportions)))

# Create legend labels with proportion and adjust the legend font size
legend_labels <- paste(co2_capacity_totals.Basin$Basin, " (", round(proportions * 100, 1), "%)", sep = "")
cex_value <- 0.5  # Adjust the font size (change as needed)

# Add a legend in the top-right corner
legend("bottomright", legend = legend_labels, fill = rainbow(length(proportions)), bty = "n", title = "Basin", cex = cex_value)

# Boxplot of co2capacity against on.off.shore
ggplot(ccs.dt, aes(x = On.off.shore, y = CO2.storage.capacity)) +
  geom_boxplot() +
  labs(title = "On vs Offshore Visualization",
       x = "On vs Offshore",
       y = "CO2 Storage Capacity")
#on.off.shore median were clearly different & no significant overlap of IQR, may be a good categorical predictor

# Basin against CO2 Storage visualization
ggplot(ccs.dt, aes(x = Basin, y = CO2.storage.capacity)) +
  geom_boxplot() +
  labs(title = "Basin against CO2 Storage Visualization",
       x = "Basin",
       y = "CO2 Storage Capacity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability
#many medians of "Basin" categories overlap with the IQR (Interquartile Range) of RAKB
#it suggests that the "Basin" variable may not be a strong categorical predictor

ggplot(ccs.dt, aes(x = Formation, y = CO2.storage.capacity)) +
  geom_boxplot() +
  labs(title = "Formation against CO2 Storage Visualization",
       x = "Formation",
       y = "CO2 Storage Capacity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# the spread and the median are significantly different, consider to be a good categorical variable 


ggplot(ccs.dt, aes(x = Depositional.environment, y = CO2.storage.capacity)) +
  geom_boxplot() +
  labs(title = "Depositional environment against CO2 Storage Visualization",
       x = "Depositional environment",
       y = "CO2 Storage Capacity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# not a good categorical variable 

ggplot(ccs.dt, aes(x = Lithology, y = CO2.storage.capacity)) +
  geom_boxplot() +
  labs(title = "Lithology environment against CO2 Storage Visualization",
       x = "Lithology environment",
       y = "CO2 Storage Capacity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# median is the same,  not a good categorical variable 

# Scatterplot for temperature against pressure
ggplot(ccs.dt, aes(x = Temperature, y = Pressure)) +
  geom_point() +
  labs(title = "Scatterplot: Pressure vs Temperature",
       x = "Temperature",
       y = "Pressure")

# Scatterplot for pressure against CO2 density
ggplot(ccs.dt, aes(x =  CO2.density , y =Pressure)) +
  geom_point() +
  labs(title = "Scatterplot: Pressure vs CO2 Density",
       x = "CO2 Density",
       y = "Pressure")

#----------------------------- Correlation between all continuous variables------------------------

#Calculating correlation
cont.variables <- c("Area", "Gross.thickness", "Net.to.Gross.thickness", "Porosity", "CO2.density", "Pressure", "Temperature", "CO2.storage.capacity")
cont.data <- ccs.dt[, ..cont.variables]
cor_matrix <- round(cor(cont.data), digits = 5)

# Create a correlation plot
corrplot(cor_matrix, method = "color", type = "full", tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7)
## temperature & pressure & porosity & CO2.density are highly correlated

# Create a matrix of box plots
pairs(ccs.dt[, ..cont.variables], main = "Box Plots of Continuous Variables", pch = 20, col = "skyblue")
## the scatter plot of porosity against co2 density, pressure, temperature is similar
## vif check are required

#-----------------------------Correlation betwen all categorical variable-----------------------------

# Load the necessary packages
library(vcd)

# Create a list of all categorical variables
categorical_vars <- c("Basin", "Formation", "Lithology", "Depositional.environment", "On.off.shore")

# Initialize an empty data frame to store results
result_table <- data.frame(Pair = character(0), P_Value = numeric(0))

# Generate all pairs of categorical variables
for (i in 1:(length(categorical_vars) - 1)) {
  for (j in (i + 1):length(categorical_vars)) {
    var1 <- categorical_vars[i]
    var2 <- categorical_vars[j]
    
    # Create a contingency table for the selected pair of variables
    contingency_table <- table(ccs.dt[, get(var1)], ccs.dt[, get(var2)])
    
    # Perform Fisher's Exact Test with simulation
    fisher_test_result <- fisher.test(contingency_table, simulate.p.value = TRUE)
    
    # Store the results in the data frame
    result_table <- rbind(result_table, data.frame(Pair = paste(var1, var2, sep = "_"), P_Value = fisher_test_result$p.value))
  }
}

# Print the table of p-values
print(result_table)
## Basin & Formation & Depositional.environment is highly correlated (p<0.05) with all the categorical variable, remove Basin & Formation & Depositional.environment


#--------------------------------------Linear regression 1 ----------------------------------------------------

excluded_vars <- c("Basin", "Depositional.environment", "Lithology", "Temperature", "CO2.density")
included_vars <- setdiff(names(ccs.dt), c("CO2.storage.capacity", excluded_vars))

# Split the dataset into training (70%) and testing (30%) sets
set.seed(2000)

# manually splitting
train_indices <- sample(seq_len(nrow(ccs.dt)), 0.7 * nrow(ccs.dt))
train_data <- ccs.dt[train_indices, ]
test_data <- ccs.dt[-train_indices, ]

train_data <- as.data.frame(train_data)
test_data <- as.data.frame(test_data)

# Run linear regression with the modified design matrix
lm_model <- lm(CO2.storage.capacity ~ ., data = train_data[, c(included_vars, "CO2.storage.capacity")])


# Summary of the linear regression model
summary(lm_model)

# Make predictions on the training set
train_predictions <- predict(lm_model, newdata = train_data)

# Make predictions on the test set
test_predictions <- predict(lm_model, newdata = test_data)

# Calculate RMSE for training set
train_rmse <- sqrt(mean((train_predictions - train_data$CO2.storage.capacity)^2))
print(paste("Training RMSE:", train_rmse))
# "Training RMSE: 1.36340313175242"

# Calculate RMSE for the test set
test_rmse <- sqrt(mean((test_predictions - test_data$CO2.storage.capacity)^2))
print(paste("Test RMSE:", test_rmse))
# "Test RMSE: 2.2989374340046"

RMSE.summary.LR1 <- data.table(
  Dataset = c("Test Set", "Training Set"),
  RMSE = c(test_rmse, train_rmse)
)
RMSE.summary.LR1

#------------------------------------------------------Linear regression 2 ------------------------------------------------
# Fit a linear regression model

set.seed(2000)
#train-test split
train <- sample.split(Y = ccs.dt$CO2.storage.capacity, SplitRatio = 0.7)
trainset <- subset(ccs.dt, train == T)
testset <- subset(ccs.dt, train == F)

lm_model <- lm(CO2.storage.capacity ~. - Basin - Depositional.environment - Lithology, data = trainset)
summary(lm_model)
vif(lm_model)

#Drop CO2.density and Pressure
lm_model2 <- lm(CO2.storage.capacity ~. - Basin - Depositional.environment - Lithology -CO2.density -Pressure, data = trainset)
summary(lm_model2)
vif(lm_model2)

#Drop Porosity
lm_model3 <- lm(CO2.storage.capacity ~. - Basin - Depositional.environment - Lithology -CO2.density -Pressure -Porosity, data = trainset)
summary(lm_model3)
vif(lm_model3)

#Drop Temperature
lm_model4 <- lm(CO2.storage.capacity ~. - Basin - Depositional.environment - Lithology - CO2.density - Pressure - Porosity - Temperature, data = trainset)
summary(lm_model4)
vif(lm_model4)

#Drop Gross.thickness
lm_model5 <- lm(CO2.storage.capacity ~. - Basin - Depositional.environment - Lithology - CO2.density - Pressure - Porosity - Temperature - Gross.thickness, data = trainset)
summary(lm_model5)
vif(lm_model5)

#Drop Net.to.Gross.Thickness
lm_model6 <- lm(CO2.storage.capacity ~ On.off.shore + Formation + Area, data = trainset)
summary(lm_model6)
vif(lm_model6)

# Predict on the training and testing sets
train_preds <- predict(lm_model6, newdata = trainset)

# Omit values equal to "A"
testset <- subset(testset, Formation %in% levels(trainset$Formation))
new_levels <- c("Al Wajh (Graben)", "Al Wajh, Pre-rift", "Burqan-Maqna")
testset <- subset(testset, Formation != new_levels)
test_preds <- predict(lm_model6, newdata = testset)

# Calculate error metrics (e.g., Mean Squared Error)
train_rmse <- sqrt(mean((train_preds - trainset$CO2.storage.capacity)^2))
test_rmse <- sqrt(mean((test_preds - testset$CO2.storage.capacity)^2))

matrix.rep <- matrix( c(train_rmse, test_rmse), byrow = TRUE)
colnames(matrix.rep) <- "Linear Regression RMSE:"
rownames(matrix.rep) <- c("Training", "Testing")
print(matrix.rep)

#Diagnostic Graphs based on optimised model (lm_model6)
par(mfrow = c(2,2))
plot(lm_model6)
par(mfrow = c(1,1))

#-----------------------------------------------CART-------------------------------------------------------------

#Growing Tree to max
cart1 <- rpart(CO2.storage.capacity ~ ., data = trainset, method = 'anova' , control = rpart.control(minsplit = 2, cp = 0))
printcp(cart1)
plotcp(cart1)
print(cart1)
rpart.plot(cart1, nn = T, main = "Maximum Tree")

#Pruning the Tree...

#Finding the most Optimal split
# Compute min CVerror + 1SE in maximal tree cart1.
CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] + cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]
CVerror.cap
# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart1.
i <- 1; j<- 4
while (cart1$cptable[i,j] > CVerror.cap) {i <- i + 1}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
optimalcp = ifelse(i > 1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]), 1)
optimalcp
#Prune!
optimalcart <- prune(cart1, cp = optimalcp)
printcp(optimalcart)
plotcp(optimalcart)
print(optimalcart)
rpart.plot(optimalcart, nn = T, main = "Pruned Tree")
optimalcart$variable.importance
summary(optimalcart)

# Train set Error
MSE <- 21.345 * 0.032560
RMSE <- sqrt(MSE)
print(RMSE)


#Applying model to testset, checking model error
optimalcart.predict <- predict(optimalcart, newdata = testset)
MSE2 <- mean((optimalcart.predict - testset$CO2.storage.capacity)^2)
RMSE2 <- sqrt(MSE2)
print(RMSE2)

