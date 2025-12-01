library(dplyr)
library(tidyr)
library(arrow)

fitness <- read.csv('./health_fitness_dataset.csv')

summary(fitness)

## data pre-processing
fitness_clean <- fitness %>%
  mutate(
    health_condition = ifelse(is.na(health_condition) | health_condition == "",
                              "no illness",
                              health_condition)
  )
fitness_clean$participant_id <- as.factor(fitness_clean$participant_id)
fitness_clean$date <- as.Date(fitness_clean$date)
fitness_clean$gender <- as.factor(fitness_clean$gender)
fitness_clean$activity_type <- as.factor(fitness_clean$activity_type)

fitness_clean$smoking_status <- as.factor(fitness_clean$smoking_status)
fitness_clean$health_condition <- as.factor(fitness_clean$health_condition)
fitness_clean$intensity <- as.factor(fitness_clean$intensity)
fitness_clean <- fitness_clean[fitness_clean$daily_steps >=0, ]
summary(fitness_clean)

unique(fitness_clean$activity_type)
unique(fitness_clean$gender)

saveRDS(fitness_clean, "./output/fitness.rds")
write_parquet(fitness_clean, "./output/fitness.parquet")

