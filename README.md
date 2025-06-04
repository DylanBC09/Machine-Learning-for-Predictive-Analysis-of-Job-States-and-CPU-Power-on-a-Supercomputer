---
title: "Machine Learning for Predictive Analysis of CPU
Power and Job States on a Supercomputer"
author: "Dylan Benavides"
date: "2025-05-24"
output: html_document
---

# Libraries

```{r}
source("Functions.R")
library(scales)
library(readr)
library(dplyr)
library(lubridate)
library(datawizard)
library(GGally)
library(ggcorrplot)
library(caret)
library(traineR)
library(randomForest)
library(vcd)
library(stringr)
library(scales)
library(ggplot2)
library(grid)
```

El primer modelo busca dar respuesta a la pregunta: 
¿Conociendo \texttt{ReqCPUS}, \texttt{ReqMem}, \texttt{ReqNodes}, \texttt{ResvCPURAW}, \texttt{TimelimitRaw}, \texttt{Priority}, \texttt{SubmitHour}, \texttt{SubmitWeekday}, \texttt{Partition} y \texttt{QOS} de un trabajo enviado al cluster, es posible anticipar si dicho trabajo se completará o va a fallar?

# Modelo 1. Clasificar el estado final de un trabajo

```{r}
data <- read.csv("dataset.csv", sep = "|")
data <- data %>% mutate(State=gsub("CANCELLED by \\d+$", "CANCELLED", State))
data <- data[c("ConsumedEnergyRaw","CPUTimeRAW","ReqCPUS",
               "ReqMem","ReqNodes","ResvCPURAW",
               "Submit","TimelimitRaw","Partition", 
               "Priority","QOS","State")]

data$Submit <- ymd_hms(data$Submit)
data$SubmitHour <- hour(data$Submit)
data$SubmitWeekday <- wday(data$Submit)

set1 <- data %>% filter(Partition=="andalan" | Partition=="andalan-debug" | Partition=="andalan-long")
set2 <- data %>% filter(Partition=="dribe" | Partition=="dribe-long" | Partition=="dribe-debug" | Partition=="dribe-test")
set3 <- data %>% filter(Partition=="nu" | Partition=="nu-all" | Partition=="nu-debug" | Partition=="nu-long" | Partition=="nu-wide")
set4 <- data %>% filter(Partition=="nukwa" | Partition=="nukwa-debug" | Partition=="nukwa-long" | Partition=="nukwa-v100" | Partition=="nukwa-wide")
set5 <- data %>% filter(Partition=="kura" | Partition=="kura-all" | Partition=="kura-debug" | Partition=="kura-long" | Partition == "kura-test" | Partition =="kura-wide")

sets <- list(set1, set2, set3, set4, set5)
names(sets) <- c("Andalan", "Dribe", "Nu", "Nukwa", "Kura")
row.counts <- sapply(sets, nrow) #cantidad de trabajos por partición.

row.counts <- data.frame(Trabajos=row.counts)
row.counts$P <- rownames(row.counts)
ggplot(row.counts, aes(x = P, y = Trabajos, fill=P)) + geom_col(color = "black") + theme_minimal()
```

## Modelo de clasificación

```{r}
data.clasificacion <- data %>% select(-c("ConsumedEnergyRaw", "CPUTimeRAW", "Submit"))

data.clasificacion <- data.clasificacion %>%
  mutate(Partition = case_when(
    str_detect(Partition, "^andalan") ~ "andalan",
    str_detect(Partition, "^dribe") ~ "dribe",
    str_detect(Partition, "^nukwa") ~ "nukwa",
    str_detect(Partition, "^nu") ~ "nu",
    str_detect(Partition, "^kura") ~ "kura",
    TRUE ~ Partition  
  ))

data.clasificacion <- data.clasificacion %>% filter(Partition != "nukwa")

data.clasificacion <- data.clasificacion %>% filter(State != "CANCELLED" & State != "NODE_FAIL" & State != "TIMEOUT")
data.clasificacion$State <- as.factor(data.clasificacion$State)
data.clasificacion$Partition <- as.factor(data.clasificacion$Partition)
data.clasificacion$QOS <- as.factor(data.clasificacion$QOS)
data.clasificacion$ReqMem <- sapply(data.clasificacion$ReqMem, memory.function)

data.clasificacion$State <- as.factor(data.clasificacion$State)
data.clasificacion$QOS <- as.factor(data.clasificacion$QOS)
data.clasificacion$Partition <- as.factor(data.clasificacion$Partition)
#data.clasificacion$SubmitHour <- as.factor(data.clasificacion$SubmitHour)
#data.clasificacion$SubmitWeekday <- as.factor(data.clasificacion$SubmitWeekday)


data.clasificacion$TimelimitRaw[data.clasificacion$TimelimitRaw=="Partition_Limit"] <- NA
partitionlimit <- colSums(is.na(data.clasificacion)) #Trabajos con partition limit
data.clasificacion$TimelimitRaw[is.na(data.clasificacion$TimelimitRaw)] <- max(data.clasificacion$TimelimitRaw, na.rm = TRUE)

data.clasificacion$TimelimitRaw <- as.numeric(data.clasificacion$TimelimitRaw) 
data.clasificacion$TimelimitRaw <- data.clasificacion$TimelimitRaw*60

data.clasificacion$ResvCPURAW <- winsorize(data.clasificacion$ResvCPURAW, threshold = 0.05)
data.clasificacion$Priority <- winsorize(data.clasificacion$Priority, threshold = 0.05)
data.clasificacion$ReqMem <- winsorize(data.clasificacion$ReqMem, threshold = 0.05)
data.clasificacion$ReqCPUS <- winsorize(data.clasificacion$ReqCPUS, threshold = 0.05)
data.clasificacion$TimelimitRaw <- winsorize(data.clasificacion$TimelimitRaw, threshold = 0.05)

#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
data.clasificacion$CPUs.per.node<- data.clasificacion$ReqCPUS / data.clasificacion$ReqNodes
data.clasificacion$Mem.per.CPU <- data.clasificacion$ReqMem / data.clasificacion$ReqCPUS
data.clasificacion$CPU.time.requested <- data.clasificacion$ReqCPUS * data.clasificacion$TimelimitRaw

data.clasificacion$Mem.per.CPU <- log(data.clasificacion$Mem.per.CPU)
data.clasificacion$CPU.time.requested <- log(data.clasificacion$CPU.time.requested)
#data.clasificacion$ResvCPURAW <- log(data.clasificacion$ResvCPURAW)

data.clasificacion <- data.clasificacion %>% select(-c("ReqCPUS", "ReqNodes", "TimelimitRaw","ReqMem"))
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

data.clasificacion.num <- data.clasificacion[sapply(data.clasificacion, is.numeric)]
data.clasificacion.cat <- data.clasificacion[sapply(data.clasificacion, is.factor)]

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

data.clasificacion.num$ResvCPURAW <- min_max_norm(data.clasificacion.num$ResvCPURAW)
data.clasificacion.num$Priority <- min_max_norm(data.clasificacion.num$Priority)
data.clasificacion.num$CPUs.per.node <- min_max_norm(data.clasificacion.num$CPUs.per.node)
data.clasificacion.num$Mem.per.CPU <- min_max_norm(data.clasificacion.num$Mem.per.CPU)
data.clasificacion.num$CPU.time.requested <- min_max_norm(data.clasificacion.num$CPU.time.requested)
data.clasificacion.num$SubmitHour <- min_max_norm(data.clasificacion.num$SubmitHour)
data.clasificacion.num$SubmitWeekday <- min_max_norm(data.clasificacion.num$SubmitWeekday)

data.clasificacionf <- data.frame(data.clasificacion.num, data.clasificacion.cat)


Corrc(data.clasificacionf)

```

#Modelo.

Note que COMPLETED: 13,285 (≈ 36%)
FAILED:    23,445 (≈ 64%) Al ser COMPLETED la minoritaria, es coherente marcarla como positiva para poder aplicar estrategias de balanceo si fuera necesario (como SMOTE, submuestreo, etc.), además de cuidar las métricas como el recall de COMPLETED.


```{r}
# Tus gráficos
g1 <- Corrc(data.clasificacionf)[[17]]
g2 <- Corrc(data.clasificacionf)[[18]]
g3 <- Corrc(data.clasificacionf)[[19]]
g4 <- Corrc(data.clasificacionf)[[20]]
g5 <- Corrc(data.clasificacionf)[[21]]
g6 <- Corrc(data.clasificacionf)[[22]]
g7 <- Corrc(data.clasificacionf)[[23]]

# Definir un tema base con texto grande
tema_grande <- theme(
  text = element_text(size = 16),               # Texto general
  axis.title = element_text(size = 18),         # Títulos de ejes
  axis.text = element_text(size = 14),          # Texto en ejes
  legend.text = element_text(size = 16),        # Texto leyenda
  legend.title = element_text(size = 18, face = "bold"),  # Título leyenda
  plot.title = element_text(size = 20, face = "bold")     # Título gráfico (si hay)
)

# Aplicar a todos los gráficos antes de quitar leyenda (para que la leyenda también aumente)
g1 <- g1 + tema_grande + theme(legend.position = "none")
g2 <- g2 + tema_grande + theme(legend.position = "none")
g3 <- g3 + tema_grande + theme(legend.position = "none")
g4 <- g4 + tema_grande + theme(legend.position = "none")
g5 <- g5 + tema_grande + theme(legend.position = "none")
g6 <- g6 + tema_grande + theme(legend.position = "none")

# En g7 aumentamos texto y mantenemos leyenda para extraerla
g7 <- g7 + tema_grande + theme(
  legend.position = "top",
  legend.direction = "horizontal"
)

# El resto del código queda igual
library(ggplot2)
library(gridExtra)
library(grid)



# Tema para texto grande
tema_grande <- theme(
  text = element_text(size = 16),
  axis.title = element_text(size = 18),
  axis.text = element_text(size = 14),
  legend.text = element_text(size = 16),
  legend.title = element_text(size = 18, face = "bold"),
  plot.title = element_text(size = 20, face = "bold")
)

# Aplicar tema y ocultar leyendas donde corresponde
g1 <- g1 + tema_grande + theme(legend.position = "none")
g2 <- g2 + tema_grande + theme(legend.position = "none")
g3 <- g3 + tema_grande + theme(legend.position = "none")
g4 <- g4 + tema_grande + theme(legend.position = "none")
g5 <- g5 + tema_grande + theme(legend.position = "none")
g6 <- g6 + tema_grande + theme(legend.position = "none")

# Para g7: texto grande y leyenda horizontal arriba
g7 <- g7 + tema_grande + theme(
  legend.position = "top",
  legend.direction = "horizontal"
)

# Extraer la leyenda de g7
get_legend <- function(myggplot) {
  tmp <- ggplotGrob(myggplot)
  leg_index <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg_index]]
  return(legend)
}
legend <- get_legend(g7)

# Quitar leyenda de g7 para ponerla aparte
g7 <- g7 + theme(legend.position = "none")

# Crear espacios vacíos para centrar último gráfico
empty1 <- nullGrob()
empty2 <- nullGrob()

# Organizar gráficos en grilla
plots_grid <- arrangeGrob(
  g1, g2, g5,
  g6, g7, g3,
  empty1, g4, empty2,
  ncol = 3
)

# Combinar leyenda arriba + grilla
final_plot <- arrangeGrob(
  legend,
  plots_grid,
  ncol = 1,
  heights = c(1, 10)
)

# Mostrar en RStudio
grid.newpage()
grid.draw(final_plot)

# Guardar PDF con texto grande
pdf("holi.pdf", width = 14, height = 10)
grid.draw(final_plot)
dev.off()

```

## models


```{r}
data.clasificacionf <- data.clasificacionf %>% select(-ResvCPURAW, -SubmitWeekday)

# Reordená los niveles para que "COMPLETED" sea la clase positiva
data.clasificacionf$State <- relevel(data.clasificacionf$State, ref = "FAILED")

graficos <- Modelsc(data.clasificacion)
```

```{r}
library(ggplot2)
library(patchwork)

#graficos <- Modelsc(data.clasificacionf)

colores <- rainbow(6)
nombres_modelos <- c("SVM", "KNN", "RF", "NNET", "XGBOOSTING", "GLM")

# Extraer gráficos
im2 <- graficos[[2]]  # Overall Accuracy
im4 <- graficos[[4]]  # Failed category accuracy
im6 <- graficos[[6]]  # Completed category accuracy

# Ajustar im2: leyenda visible solo aquí
im2 <- im2 +
  scale_fill_manual(values = colores, name = "Model") +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.title = element_text(size = 18, face = "bold"),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  ) +
  coord_cartesian(ylim = c(0, max(im2$data$precision) * 1.15)) +
  labs(title = NULL, x=NULL)

# Ajustar im4: leyenda oculta completamente
im4 <- im4 +
  scale_fill_manual(values = colores, guide = "none") +  # OJO: guiar = "none" elimina leyenda
  theme(
    legend.position = "none",
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  ) +
  coord_cartesian(ylim = c(0, max(im4$data$failed) * 1.15)) +
  labs(title = NULL,x=NULL)

# Ajustar im6: leyenda oculta completamente
im6 <- im6 +
  scale_fill_manual(values = colores, guide = "none") +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  ) +
  coord_cartesian(ylim = c(0, max(im6$data$completed) * 1.15)) +
  labs(title = NULL,x=NULL)

# Combinar gráficos
combinado <- im2 / im4 / im6 + plot_layout(guides = "collect") &
  theme(
    legend.position = "top",
    legend.direction = "horizontal"
  )

# Guardar PDF
ggsave("combinado_metricas.pdf", combinado, width = 10, height = 12)

```



## Umbral de probabilidad modelo Random Forest

Lo siguiente está correcto.


```{r}
muestra  <- createDataPartition(y = data.clasificacionf$State, p = 0.8, list = F)
taprendizaje <- data.clasificacionf[muestra, ]
ttesting     <- data.clasificacionf[-muestra, ]  
taprendizaje$State <- factor(taprendizaje$State, levels = c("FAILED", "COMPLETED"))
ttesting$State <- factor(ttesting$State, levels = c("FAILED", "COMPLETED"))

modeloA <- train.randomForest(State ~ ., data = taprendizaje, probability=T)
prediccionA <- predict(modeloA, ttesting)

mc <- confusion.matrix(ttesting, prediccionA)
general.indexes(mc = mc)
```

```{r}
prediccion <- predict(modeloA, ttesting, type = "prob") # Para que me retorne la probabilidad
Clase      <- ttesting$State
head(Clase)

Score      <- prediccion$prediction[,"COMPLETED"]
head(Score)

Corte      <- 0.5
Prediccion <- ifelse(Score > Corte, "COMPLETED", "FAILED")
MC         <- table(Clase, Pred = factor(Prediccion, levels = c("FAILED", "COMPLETED")))
general.indexes(mc = MC)


Clase <- ttesting$State
Score <- prediccion$prediction[,"COMPLETED"]
for(Corte in seq(1, 0, by = -0.05)) {
    Prediccion <- ifelse(Score >= Corte, "COMPLETED", "FAILED")
    MC         <- table(Clase, Pred = factor(Prediccion, levels = c("FAILED", "COMPLETED")))
    cat("\nCorte usado para la Probabilidad = ")
    cat(Corte)
    cat("\n")
    print(general.indexes(mc = MC))
    cat("\n========================================")
}

```


## Curva ROC

```{r}
muestra  <- createDataPartition(y = data.clasificacion$State, p = 0.8, list = F)
taprendizaje <- data.clasificacionf[muestra, ]
ttesting     <- data.clasificacionf[-muestra, ]  
taprendizaje$State <- factor(taprendizaje$State, levels = c("FAILED", "COMPLETED"))
ttesting$State <- factor(ttesting$State, levels = c("FAILED", "COMPLETED"))

modelo <- train.randomForest(State ~ ., data = taprendizaje, ntree=50)
prediccion   <- predict(modelo, ttesting, type = "prob")
head(prediccion)

Score        <- prediccion$prediction[,"COMPLETED"]
Clase        <- ttesting$State

#bosque <- randomForest(State ~., data = taprendizaje, #ntree=200, keep.forest = FALSE, importance = FALSE)
#plot(bosque, log = "y", title = "")

library(pROC)
roc_obj <- roc(ttesting$State, Score, auc = TRUE, ci = F)

pdf("curva_ROC.pdf", width = 7, height = 7)  
plot(roc_obj, 
     col = "blue", 
     xlab = "1-Specificity", 
     ylab = "Sensitivity", 
     print.auc = TRUE,   
     print.auc.cex = 2,   # tamaño del texto AUC más grande
     cex.lab = 1.5)
dev.off()



```





```{r}
muestra  <- createDataPartition(y = data.clasificacion$State, p = 0.8, list = F)
taprendizaje <- data.clasificacion[muestra, ]
ttesting     <- data.clasificacion[-muestra, ]  

taprendizaje$State <- factor(taprendizaje$State, levels = c("FAILED", "COMPLETED"))
ttesting$State <- factor(ttesting$State, levels = c("FAILED", "COMPLETED"))

modeloA <- train.randomForest(State ~ ., data = taprendizaje)
prediccionA <- predict(modeloA, ttesting)

pred1 <- predict(modeloA, ttesting, type = "prob")
head(pred1)
Estado1 <- pred1$prediction[,"COMPLETED"]

confusion.matrix(ttesting, prediccionA)
confusionMatrix(ttesting$State, prediccionA$prediction, positive = "COMPLETED")

#ROC.plot(Estado1 , ttesting$State, color = "red")
#ROC.area(Estado1,Clase1)

library(pROC)
roc_obj <- roc(ttesting$State, Estado1, auc = TRUE, ci = TRUE)
pdf("curva_ROC.pdf", width = 7, height = 7)  
plot(roc_obj, 
     col = "blue", 
     xlab = "1-Specificity", 
     ylab = "Sensitivity", 
     print.auc = TRUE,   cex.lab = 1.5)
dev.off()

```

```{r}
for(Corte in seq(1, 0, by = -0.05)) {
    Prediccion <- ifelse(Estado1 >= Corte, "FAILED", "COMPLETED")
    Pred <- factor(Prediccion, levels = c("COMPLETED", "FAILED"))
    MC <- table(Clase1, Pred)
    cat("\nCorte usado para la Probabilidad = ", Corte, "\n")
    print(general.indexes(mc = MC))
    cat("\n========================================\n")
}
```

## Umbral 0.55

```{r}
library(caret)
library(randomForest)
library(ROCR)

# Crear la partición
muestra <- createDataPartition(y = data.clasificacion$State, p = 0.8, list = FALSE)
taprendizaje <- data.clasificacion[muestra, ]
ttesting     <- data.clasificacion[-muestra, ]  

# Entrenar modelo con caret
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE)
#modeloA <- train(State ~ ., data = taprendizaje, method = "rf", trControl = control)

# Predecir probabilidades
pred1 <- predict(modeloA, ttesting, type = "prob")

# Aplicar umbral personalizado
prob_failed <- pred1[,"FAILED"]
prediccion_umbral <- ifelse(prob_failed > 0.1, "FAILED", "COMPLETED")
prediccion_umbral <- factor(prediccion_umbral, levels = c("COMPLETED", "FAILED"))

# Clases reales
Clase1 <- factor(ttesting$State, levels = c("COMPLETED", "FAILED"))

# Matriz de confusión
library(caret)

mc1 <- confusionMatrix(data = prediccion_umbral, reference = Clase1)

# Curva ROC
ROC.plot(prob_failed, Clase1, color = "red")

# Área bajo la curva
ROC.area(prob_failed, Clase1)

```


## Modelo de clasificación seleccionado: Random Forest

```{r}
modelo1 <- train.randomForest(State ~ ., data = taprendizaje)
modelo1

pred1 <- predict(modelo1, ttesting, type = "prob")
Estado1 <- pred1$prediction[,"FAILED"]
Clase1 <- factor(ttesting$State, levels = c("COMPLETED", "FAILED"))
ROC.plot(Estado1,Clase1)
ROC.area(Estado1,Clase1)
```

Ahora que tengo un modelo que anticipa si un trabajo se completará o fallará, entonces puedo determinar la potencia promedio de CPU que se obtendrá de un trabajo que el modelo 1 clasifique como completado. 














# Modelo 2. Predecir CPUPower de un trabajo que el modelo 1 clasifica como completado, para informar la potencia de CPU estimada que tendrá dicho Job (Modelo usando todos las particiones a la vez)

## Dataset

```{r}
data <- read.csv("dataset.csv", sep = "|")
data <- data %>% mutate(State=gsub("CANCELLED by \\d+$", "CANCELLED", State))

set1 <- data %>% filter(Partition=="andalan" | Partition=="andalan-debug" | Partition=="andalan-long")
set1 <- data %>%
  filter(Partition %in% c("andalan", "andalan-debug", "andalan-long")) %>%
  mutate(Partition = "andalan")

set2 <- data %>% filter(Partition=="dribe" | Partition=="dribe-long" | Partition=="dribe-debug" | Partition=="dribe-test")
set2 <- data %>%
  filter(Partition %in% c("dribe", "dribe-long", "dribe-debug", "dribe-test")) %>%
  mutate(Partition = "dribe")

set3 <- data %>% filter(Partition=="kura" | Partition=="kura-all" | Partition=="kura-debug" | Partition=="kura-long" | Partition == "kura-test" | Partition =="kura-wide")
set3 <- data %>%
  filter(Partition %in% c("kura", "kura-all", "kura-debug", "kura-long", "kura-test", "kura-wide")) %>%
  mutate(Partition = "kura")

set4 <- data %>% filter(Partition=="nu" | Partition=="nu-all" | Partition=="nu-debug" | Partition=="nu-long" | Partition=="nu-wide")
set4 <- data %>%
  filter(Partition %in% c("nu", "nu-all", "nu-debug", "nu-long", "nu-wide")) %>%
  mutate(Partition = "nu")

set.total <- rbind(set1, set2, set3, set4)

sets <- list(set1, set2, set3, set4)
jobs <- sapply(sets, nrow) 
names(jobs) <- c("Andalan", "Dribe", "Kurá", "Nu")
jobs <- as.data.frame(jobs)
```

## Procesamiento, correlación, modelado y calibración

```{r}
na.counts <- dim(set.total %>% filter(if_any(everything(), is.na)))[1]
set.total <- na.omit(set.total)
zero.counts <- colSums(set.total==0)
set.total <- set.total %>% filter(CPUTimeRAW!=0)
set.total <- set.total %>% filter(State == "COMPLETED") 

set.total <- cpupower.function(set.total, "ConsumedEnergyRaw", "CPUTimeRAW") 
set.total$TimelimitRaw <- as.numeric(set.total$TimelimitRaw)
set.total$TimelimitRaw <- set.total$TimelimitRaw*60
set.total$ReqMem <- sapply(set.total$ReqMem, memory.function)

set.total$Submit <- ymd_hms(set.total$Submit)
set.total$SubmitHour <- hour(set.total$Submit)
set.total$SubmitWeekday <- wday(set.total$Submit)

set.total$QOS <- as.factor(set.total$QOS)
set.total$Partition <- as.factor(set.total$Partition)
#set.total$SubmitHour <- as.factor(set.total$SubmitHour)
#set.total$SubmitWeekday <- as.factor(set.total$SubmitWeekday)
set.total <- set.total %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set.total %>% filter(set.total$ResvCPURAW!=0)
set.total$ResvCPURAW[set.total$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set.total <- set.total %>% select(-c("CPUTimeRAW", "ConsumedEnergyRaw", "Submit", "State"))

vars_to_winsor <- c("CPUPower", "Priority", "ReqMem", "TimelimitRaw","ResvCPURAW")
for (var in vars_to_winsor) {
  set.total[[var]] <- winsorize(set.total[[var]], threshold = 0.15)
}

#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
set.total$CPUs.per.node<- set.total$ReqCPUS / set.total$ReqNodes
set.total$Mem.per.CPU <- set.total$ReqMem / set.total$ReqCPUS
set.total$CPU.time.requested <- set.total$ReqCPUS * set.total$TimelimitRaw


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
library(moments)
library(e1071)

asimetria <- skewness(set.total$CPUPower)
curtosis <- kurtosis(set.total$CPUPower)
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    
set.total$CPUPower <- log(set.total$CPUPower)
#set.total$CPUs.per.node <-  log(set.total$CPUs.per.node)
set.total$Mem.per.CPU <- log(set.total$Mem.per.CPU)
set.total$CPU.time.requested <- log(set.total$CPU.time.requested)
set.total$ResvCPURAW <- log(set.total$ResvCPURAW)

set.total <- set.total %>% select(-c("ReqCPUS", "ReqNodes", "TimelimitRaw","ReqMem"))
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

set.totalf <- set.total %>% select_if(is.factor)
set.totaln <- set.total %>% select_if(is.numeric) 

set.totaln$CPUPower <- min_max_norm(set.totaln$CPUPower)
set.totaln$ResvCPURAW <- min_max_norm(set.totaln$ResvCPURAW)
set.totaln$Priority <- min_max_norm(set.totaln$Priority)
set.totaln$CPUs.per.node <- min_max_norm(set.totaln$CPUs.per.node)
set.totaln$Mem.per.CPU <- min_max_norm(set.totaln$Mem.per.CPU)
set.totaln$CPU.time.requested <- min_max_norm(set.totaln$CPU.time.requested)
set.totaln$SubmitHour <- min_max_norm(set.totaln$SubmitHour)
set.totaln$SubmitWeekday <- min_max_norm(set.totaln$SubmitWeekday)


set.totaln <- set.totaln %>% select(CPUPower, everything())
set.totalfnc <- cbind(set.totalf,set.totaln)

#ggpairs(set.totaln)

corr(set.totaln)

#modeling <- models(set.totalfnc)

#calibration.knn(set.totalfnc)

#calibration.RF(set.totalfnc)


#library(ggplot2)

#pdf("maplot.pdf", width = 7, height = 7)

#print(
 # modeling[[1]] + 
  #  theme(
   #   axis.title.x = element_text(size = 20),  # tamaño del texto eje x
    #  axis.title.y = element_text(size = 20)   # tamaño del texto eje y
#    )
#)

#dev.off()


```

## Modelo predictivo seleccionado: Random Forest

```{r}
#Cantidad de árboles:

#bosque <- randomForest(CPUPower ~., data = set.totalfnc, ntree=200, keep.forest = FALSE, importance = #FALSE)
#plot(bosque, log = "y", title = "")

muestra  <- createDataPartition(y = set.totalfnc$CPUPower, p = 0.8, list = F)
ttraining <- set.totalfnc[muestra, ]
ttesting     <- set.totalfnc[-muestra, ]  

modelo <- randomForest(CPUPower ~ ., ttraining, nodesize = 1, mtry=3, ntree = 100, importance = TRUE)
predicciones <- predict(modelo, ttesting)

errores <- accuracy.metrics(predicciones,ttesting$CPUPower, cantidad.variables.predictoras=(ncol(ttesting)-1))

errores


#plot.real.prediccion(predicciones, ttesting$CPUPower)
#modelo2
importance(modelo)
varImpPlot(modelo)


# Abrir dispositivo PDF
pdf("prediccion_vs_real.pdf", width = 8, height = 6)

# Generar y personalizar el gráfico
plot.real.prediccion(predicciones, ttesting$CPUPower) +
 theme(
  axis.title.x = element_text(size = 16),
 axis.title.y = element_text(size = 16)
  )

# Cerrar el dispositivo
dev.off()

actual <- ttesting$CPUPower
rss <- sum((actual - predicciones)^2)               # Residual Sum of Squares
tss <- sum((actual - mean(actual))^2)       # Total Sum of Squares
r2 <- 1 - rss/tss                           # R²

```   


```{r}
calibration.RF(set.totalfnc)
```

```{r}
bosque <- randomForest(CPUPower ~., data = set.totalfnc, ntree=100, keep.forest = FALSE, importance = FALSE)

```

```{r}
muestra  <- createDataPartition(y = set.totalfnc$CPUPower, p = 0.8, list = F)
ttraining <- set.totalfnc[muestra, ]
ttesting     <- set.totalfnc[-muestra, ]  


modelo  <- train.svm(CPUPower~., data = ttraining, kernel = "radial")
prediccion  <- predict(modelo, ttesting)

    
actual <- ttesting$CPUPower
rss <- sum((actual - prediccion$prediction)^2)               # Residual Sum of Squares
tss <- sum((actual - mean(actual))^2)       # Total Sum of Squares
r2 <- 1 - rss/tss                           # R²

# Mostrar el resultado
r2

``` 

```{r}
muestra  <- createDataPartition(y = set.totalfnc$CPUPower, p = 0.8, list = F)
ttraining <- set.totalfnc[muestra, ]
ttesting     <- set.totalfnc[-muestra, ]  

modelo      <- train.knn(CPUPower~., data = ttraining, kmax = 37)
prediccion  <- predict(modelo, ttesting)

actual <- ttesting$CPUPower
rss <- sum((actual - prediccion$prediction)^2)               # Residual Sum of Squares
tss <- sum((actual - mean(actual))^2)       # Total Sum of Squares
r2 <- 1 - rss/tss                           # R²

```   

```{r}
muestra  <- createDataPartition(y = set.totalfnc$CPUPower, p = 0.8, list = F)
ttraining <- set.totalfnc[muestra, ]
ttesting     <- set.totalfnc[-muestra, ]  


modelo     <- train.nnet(CPUPower~., data =ttraining , size = 100, MaxNWts = 5000, 
                               rang = 0.01, decay = 5e-4, maxit = 45, trace = TRUE)
prediccion <- predict(modelo, ttesting)

actual <- ttesting$CPUPower
rss <- sum((actual - prediccion$prediction)^2)               # Residual Sum of Squares
tss <- sum((actual - mean(actual))^2)       # Total Sum of Squares
r2 <- 1 - rss/tss 

      
```
  

```{r}
muestra  <- createDataPartition(y = set.totalfnc$CPUPower, p = 0.8, list = F)
ttraining <- set.totalfnc[muestra, ]
ttesting     <- set.totalfnc[-muestra, ]  


modelo     <- train.xgboost(CPUPower~., data = ttraining, nrounds=30)
prediccion <- predict(modelo, ttesting)
      
actual <- ttesting$CPUPower
rss <- sum((actual - prediccion$prediction)^2)               # Residual Sum of Squares
tss <- sum((actual - mean(actual))^2)       # Total Sum of Squares
r2 <- 1 - rss/tss 
      
```


```{r}
muestra  <- createDataPartition(y = set.totalfnc$CPUPower, p = 0.8, list = F)
ttraining <- set.totalfnc[muestra, ]
ttesting     <- set.totalfnc[-muestra, ]  


      modelo <- lm(CPUPower~., data = ttraining)
      prediccion <- predict(modelo, ttesting)
       
actual <- ttesting$CPUPower
rss <- sum((actual - prediccion)^2)               # Residual Sum of Squares
tss <- sum((actual - mean(actual))^2)       # Total Sum of Squares
r2 <- 1 - rss/tss 
      
```

    
# Modelo 3. Predecir CPUPower por partición de un trabajo que el modelo 1 clasifica como completado, para informar la potencia de CPU estimada que tendrá dicho Job en la partición específica (Un modelo para cada partición)

```{r}
data <- read.csv("dataset.csv", sep = "|")
data <- data %>% mutate(State=gsub("CANCELLED by \\d+$", "CANCELLED", State))

set1 <- data %>% filter(Partition=="andalan" | Partition=="andalan-debug" | Partition=="andalan-long")

set2 <- data %>% filter(Partition=="dribe" | Partition=="dribe-long" | Partition=="dribe-debug" | Partition=="dribe-test")

set3 <- data %>% filter(Partition=="kura" | Partition=="kura-all" | Partition=="kura-debug" | Partition=="kura-long" | Partition == "kura-test" | Partition =="kura-wide")

set4 <- data %>% filter(Partition=="nu" | Partition=="nu-all" | Partition=="nu-debug" | Partition=="nu-long" | Partition=="nu-wide")
```

## Set1: Andalan

```{r eval = FALSE}
na.counts <- dim(set1 %>% filter(if_any(everything(), is.na)))[1]
set1 <- na.omit(set1)
zero.counts <- colSums(set1==0)
set1 <- set1 %>% filter(CPUTimeRAW!=0)
set1 <- set1 %>% filter(State == "COMPLETED" | State == "FAILED") 
job.states <- summary(factor(set1$State))

  
set1 <- cpupower.function(set1, "ConsumedEnergyRaw", "CPUTimeRAW") 
set1$TimelimitRaw <- as.numeric(set1$TimelimitRaw)
set1$TimelimitRaw <- set1$TimelimitRaw*60
set1 <- select(set1, -ReqMem, -ReqNodes, -ConsumedEnergyRaw, -Partition, -CPUTimeRAW, -State)
set1$Submit <- as_datetime(set1$Submit)
set1$Submit <- as.numeric(set1$Submit) 
set1$QOS <- as.factor(set1$QOS)
set1 <- set1 %>% select(CPUPower, everything())
cpu.time.reserved.nonzero <- set1 %>% filter(set1$ResvCPURAW!=0)
set1$ResvCPURAW[set1$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set1$CPUPower <- winsorize(set1$CPUPower, threshold = 0.05)

set1$ResvCPURAW <- winsorize(set1$ResvCPURAW, threshold = 0.1)

set1$TimelimitRaw <- winsorize(set1$TimelimitRaw, threshold = 0.05)

set1$Priority <- winsorize(set1$Priority, threshold = 0.05)

ggpairs(set1)

set1f <- set1 %>% select_if(is.factor)
set1n <- set1 %>% select_if(is.numeric) 
set1n <- as.data.frame(scale(set1n, center=T, scale=T)) 
set1n <- set1n %>% select(CPUPower, everything())
set1fn <- cbind(set1f, set1n)

corr(set1n)
set.seed(1234)
models(set1fn)
```

## Set1: Andalan with Completed Jobs

Hay $1089$ trabajos completados.

```{r}
na.counts <- dim(set1 %>% filter(if_any(everything(), is.na)))[1]
set1 <- na.omit(set1)
zero.counts <- colSums(set1==0)
set1 <- set1 %>% filter(CPUTimeRAW!=0)
set1 <- set1 %>% filter(State == "COMPLETED") 

set1 <- cpupower.function(set1, "ConsumedEnergyRaw", "CPUTimeRAW") 
set1$TimelimitRaw <- as.numeric(set1$TimelimitRaw)
set1$TimelimitRaw <- set1$TimelimitRaw*60
set1$ReqMem <- sapply(set1$ReqMem, memory.function)

set1$Submit <- ymd_hms(set1$Submit)
set1$SubmitHour <- hour(set1$Submit)
set1$SubmitWeekday <- wday(set1$Submit)

set1$QOS <- as.factor(set1$QOS)
set1$Partition <- as.factor(set1$Partition)
set1$SubmitHour <- as.factor(set1$SubmitHour)
set1$SubmitWeekday <- as.factor(set1$SubmitWeekday)
set1 <- set1 %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set1 %>% filter(set1$ResvCPURAW!=0)
set1$ResvCPURAW[set1$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set1 <- set1 %>% select(-c("CPUTimeRAW", "ConsumedEnergyRaw", "State", "Submit", "SubmitHour"))

vars_to_winsor <- c("CPUPower", "ReqMem", "Priority", "ReqMem", "TimelimitRaw","ResvCPURAW")
for (var in vars_to_winsor) {
  set1[[var]] <- winsorize(set1[[var]], threshold = 0.15)
}

#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
set1$Mem.per.CPU <- set1$ReqMem / set1$ReqCPUS
set1$CPU.time.requested <- set1$ReqCPUS * set1$TimelimitRaw

set1$CPUPower <- log(set1$CPUPower)
set1$Mem.per.CPU <- log(set1$Mem.per.CPU)
set1$CPU.time.requested <- log(set1$CPU.time.requested)
set1$ResvCPURAW <- log(set1$ResvCPURAW)

set1 <- set1 %>% select(-c("ReqNodes", "TimelimitRaw", "ReqMem"))
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

set1f <- set1 %>% select_if(is.factor)
set1n <- set1 %>% select_if(is.numeric) 

set1n <- as.data.frame(scale(set1n, center=T, scale=T)) 
set1n <- set1n %>% select(CPUPower, everything())
set1fnc <- cbind(set1f,set1n)

ggpairs(set1n)

corr(set1n)

set.seed(1234)
models(set1fnc)
```

## Set2: Dribe

```{r eval = FALSE}
na.counts <- dim(set2 %>% filter(if_any(everything(), is.na)))[1]
set2 <- na.omit(set2)
zero.counts <- colSums(set2==0)
cero.data <- colSums(set2==0)
set2 <- set2 %>% filter(CPUTimeRAW!=0)
set2 <- set2 %>% filter(State == "COMPLETED" | State == "FAILED") 
job.states <- summary(factor(set2$State))

set2 <- cpupower.function(set2, "ConsumedEnergyRaw", "CPUTimeRAW") 
set2$TimelimitRaw <- as.numeric(set2$TimelimitRaw)
set2$TimelimitRaw <- set2$TimelimitRaw*60
set2$ReqMem <- sapply(set2$ReqMem, memory.function)

set2 <- select(set2, -ReqNodes, -ConsumedEnergyRaw, -Partition, -CPUTimeRAW, -State)
set2$Submit <- as_datetime(set2$Submit)
set2$Submit <- as.numeric(set2$Submit) 
set2$QOS <- as.factor(set2$QOS)
set2 <- set2 %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set2 %>% filter(set2$ResvCPURAW!=0)
set2$ResvCPURAW[set2$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set2$CPUPower <- winsorize(set2$CPUPower, threshold = 0.05)
set2$ResvCPURAW <- winsorize(set2$ResvCPURAW, threshold = 0.05)
set2$TimelimitRaw <- winsorize(set2$TimelimitRaw, threshold = 0.05)
set2$Priority <- winsorize(set2$Priority, threshold = 0.05)
set2$ReqMem <- winsorize(set2$ReqMem, threshold = 0.05)

q1 <- quantile(set2$ResvCPURAW, 0.25)
q3 <- quantile(set2$ResvCPURAW, 0.75)
iqr <- q3 - q1
lim_inf <- q1 - 1.5 * iqr
lim_sup <- q3 + 1.5 * iqr
outliersr <- set2 %>% filter(set2$ResvCPURAW < lim_inf | set2$ResvCPURAW > lim_sup)
set2 <- set2 %>% filter((set2$ResvCPURAW >= lim_inf) & (set2$ResvCPURAW <= lim_sup))

set2$CPUPower <- log1p(set2$CPUPower)

#pairs(set2, lower.panel = NULL)  
#ggpairs(set2)

set2f <- set2 %>% select_if(is.factor)
set2n <- set2 %>% select_if(is.numeric) 
set2n <- as.data.frame(scale(set2n, center=T, scale=T)) 
set2n <- set2n %>% select(CPUPower, everything())
set2fn <- cbind(set2f,set2n)

#corr(set2n)

#set.seed(1234)

models(set2fn)

```

## Set2: Dribe with Completed Jobs

- Hay $4993$ trabajos completos.

```{r}
na.counts <- dim(set2 %>% filter(if_any(everything(), is.na)))[1]
set2 <- na.omit(set2)
zero.counts <- colSums(set2==0)
set2 <- set2 %>% filter(CPUTimeRAW!=0)
set2 <- set2 %>% filter(State == "COMPLETED") 

set2 <- cpupower.function(set2, "ConsumedEnergyRaw", "CPUTimeRAW") 
set2$TimelimitRaw <- as.numeric(set2$TimelimitRaw)
set2$TimelimitRaw <- set2$TimelimitRaw*60
set2$ReqMem <- sapply(set2$ReqMem, memory.function)

set2$Submit <- ymd_hms(set2$Submit)
set2$SubmitHour <- hour(set2$Submit)
set2$SubmitWeekday <- wday(set2$Submit)

set2$QOS <- as.factor(set2$QOS)
set2$Partition <- as.factor(set2$Partition)
set2$SubmitHour <- as.factor(set2$SubmitHour)
set2$SubmitWeekday <- as.factor(set2$SubmitWeekday)
set2 <- set2 %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set2 %>% filter(set2$ResvCPURAW!=0)
set2$ResvCPURAW[set2$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set2 <- set2 %>% select(-c("CPUTimeRAW", "ConsumedEnergyRaw", "Submit", "State"))

vars_to_winsor <- c("CPUPower", "ReqMem", "Priority", "ReqMem", "TimelimitRaw","ResvCPURAW")
for (var in vars_to_winsor) {
  set2[[var]] <- winsorize(set2[[var]], threshold = 0.1)
}

#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
set2$CPUs.per.node<- set2$ReqCPUS / set2$ReqNodes
set2$Mem.per.CPU <- set2$ReqMem / set2$ReqCPUS
set2$CPU.time.requested <- set2$ReqCPUS * set2$TimelimitRaw

set2$CPUPower <- log(set2$CPUPower)
set2$CPUs.per.node <-  log(set2$CPUs.per.node)
set2$Mem.per.CPU <- log(set2$Mem.per.CPU)
set2$CPU.time.requested <- log(set2$CPU.time.requested)
set2$ResvCPURAW <- log(set2$ResvCPURAW)

set2 <- set2 %>% select(-c("ReqCPUS", "ReqNodes", "TimelimitRaw", "ReqMem"))
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

set2f <- set2 %>% select_if(is.factor)
set2n <- set2 %>% select_if(is.numeric) 

set2n <- as.data.frame(scale(set2n, center=T, scale=T)) 
set2n <- set2n %>% select(CPUPower, everything())
set2fnc <- cbind(set2f,set2n)

ggpairs(set2n)

corr(set2n)

set.seed(1234)
models(set2fnc)
```

## Set3: Kurá

```{r eval = FALSE}
na.counts <- dim(set3 %>% filter(if_any(everything(), is.na)))[1]
set3 <- na.omit(set3)
zero.counts <- colSums(set3==0)
cero.data <- colSums(set3==0)
set3 <- set3 %>% filter(CPUTimeRAW!=0)
set3 <- set3 %>% filter(State == "COMPLETED" | State == "FAILED") 
job.states <- summary(factor(set3$State))

set3 <- cpupower.function(set3, "ConsumedEnergyRaw", "CPUTimeRAW") 
set3$TimelimitRaw <- as.numeric(set3$TimelimitRaw)
set3$TimelimitRaw <- set3$TimelimitRaw*60
set3$ReqMem <- sapply(set3$ReqMem, memory.function)
set3$Submit <- as_datetime(set3$Submit)
set3$Submit <- as.numeric(set3$Submit) 
set3$QOS <- as.factor(set3$QOS)
set3 <- set3 %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set3 %>% filter(set3$ResvCPURAW!=0)
set3$ResvCPURAW[set3$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set3 <- set3 %>% select(-c("CPUTimeRAW", "ConsumedEnergyRaw"))
set3$Submit <- as_datetime(set3$Submit)
set3$Submit <- as.numeric(set3$Submit) 
set3$QOS <- as.factor(set3$QOS)

set3$CPUPower <- winsorize(set3$CPUPower, threshold = 0.05)
set3$ReqCPUS <- winsorize(set3$ReqCPUS, threshold = 0.05)
set3$ResvCPURAW <- winsorize(set3$ResvCPURAW, threshold = 0.05)
set3$TimelimitRaw <- winsorize(set3$TimelimitRaw, threshold = 0.05)
set3$Priority <- winsorize(set3$Priority, threshold = 0.05)

set3f <- set3 %>% select_if(is.factor)
set3n <- set3 %>% select_if(is.numeric) 
set3n$CPUPower <- log(set3n$CPUPower)


q1 <- quantile(set3$TimelimitRaw, 0.25)
q3 <- quantile(set3$TimelimitRaw, 0.75)
iqr <- q3 - q1
lim_inf <- q1 - 1.5 * iqr
lim_sup <- q3 + 1.5 * iqr
set3 <- set3 %>% filter((set3$TimelimitRaw >= lim_inf) & (set3$TimelimitRaw <= lim_sup))


ggpairs(set3)

set3n <- as.data.frame(scale(set3n, center=T, scale=T)) 
set3n <- set3n %>% select(CPUPower, everything())
set3fn <- cbind(set3f,set3n)
eliminados <- dim(set3)[1]-dim(set3)[1] 
porcentaje <- eliminados/dim(set3)[1] 

corr(set3n)

set.seed(1234)
models(set3fn)
```

## Set3: Kurá with Completed Jobs

- Tiene $2679$ trabajos completos.

```{r}
na.counts <- dim(set3 %>% filter(if_any(everything(), is.na)))[1]
set3 <- na.omit(set3)
zero.counts <- colSums(set3==0)
set3 <- set3 %>% filter(CPUTimeRAW!=0)
set3 <- set3 %>% filter(State == "COMPLETED") 

set3 <- cpupower.function(set3, "ConsumedEnergyRaw", "CPUTimeRAW") 
set3$TimelimitRaw <- as.numeric(set3$TimelimitRaw)
set3$TimelimitRaw <- set3$TimelimitRaw*60
set3$ReqMem <- sapply(set3$ReqMem, memory.function)

set3$Submit <- ymd_hms(set3$Submit)
set3$SubmitHour <- hour(set3$Submit)
set3$SubmitWeekday <- wday(set3$Submit)

set3$QOS <- as.factor(set3$QOS)
set3$Partition <- as.factor(set3$Partition)
set3$SubmitHour <- as.factor(set3$SubmitHour)
set3$SubmitWeekday <- as.factor(set3$SubmitWeekday)
set3 <- set3 %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set3 %>% filter(set3$ResvCPURAW!=0)
set3$ResvCPURAW[set3$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set3 <- set3 %>% select(-c("CPUTimeRAW", "ConsumedEnergyRaw", "Submit", "State"))

vars_to_winsor <- c("CPUPower", "ReqMem", "Priority", "ReqMem", "TimelimitRaw","ResvCPURAW")
for (var in vars_to_winsor) {
  set3[[var]] <- winsorize(set3[[var]], threshold = 0.15)
}

#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
set3$CPUs.per.node<- set3$ReqCPUS / set3$ReqNodes
set3$Mem.per.CPU <- set3$ReqMem / set3$ReqCPUS
set3$CPU.time.requested <- set3$ReqCPUS * set3$TimelimitRaw

set3$CPUPower <- log(set3$CPUPower)
set3$CPUs.per.node <-  log(set3$CPUs.per.node)
set3$Mem.per.CPU <- log(set3$Mem.per.CPU)
set3$CPU.time.requested <- log(set3$CPU.time.requested)
set3$ResvCPURAW <- log(set3$ResvCPURAW)

set3 <- set3 %>% select(-c("ReqCPUS", "ReqNodes", "TimelimitRaw","ReqMem"))
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

set3f <- set3 %>% select_if(is.factor)
set3n <- set3 %>% select_if(is.numeric) 

set3n <- as.data.frame(scale(set3n, center=T, scale=T)) 
set3n <- set3n %>% select(CPUPower, everything())
set3fnc <- cbind(set3f,set3n)

ggpairs(set3n)

corr(set3n)

set.seed(1234)
models(set3fnc)
```

## Set4: Nu

```{r eval = FALSE}
set4 <- na.omit(set4)
cero.data <- colSums(set4==0)
set4 <- set4 %>% filter(CPUTimeRAW!=0)
set4 <- set4 %>% filter(State == "COMPLETED" | State == "FAILED") #La mayoría son fallidos.
set4 <- cpupower.function(set4, "ConsumedEnergyRaw", "CPUTimeRAW") 
#set4$CPUPower <- as.integer(set4$CPUPower) con esto, la potencia se vuelve binaria, 0 y 1
set4$TimelimitRaw <- as.numeric(set4$TimelimitRaw)
set4$TimelimitRaw <- set4$TimelimitRaw*60
set4$ReqMem <- sapply(set4$ReqMem, memory.function)
set4 <- select(set4, -Partition, -ConsumedEnergyRaw, -CPUTimeRAW, -State, -Priority)
set4$Submit <- as_datetime(set4$Submit)
set4$Submit <- as.numeric(set4$Submit) 
set4$QOS <- as.factor(set4$QOS)
set4 <- set4 %>% select(CPUPower, everything())

set4.1 <- set4 %>% filter(set4$ResvCPURAW!=0)
set4$ResvCPURAW[set4$ResvCPURAW==0] <- min(set4.1$ResvCPURAW)

set4$CPUPower <- winsorize(set4$CPUPower, threshold = 0.1)
set4$ResvCPURAW <- winsorize(set4$ResvCPURAW, threshold = 0.05)
set4$TimelimitRaw <- winsorize(set4$TimelimitRaw, threshold = 0.05)
#set4$Priority <- Winsorize(set4$Priority)

#pairs(set4, lower.panel = NULL)  
#ggpairs(set4)

set4f <- set4 %>% select_if(is.factor)
set4n <- set4 %>% select_if(is.numeric) 
set4n <- as.data.frame(scale(set4n, center=T, scale=T)) 
set4n <- set4n %>% select(CPUPower, everything())
set4fn <- cbind(set4f,set4n)

#corr(set4n)

#set.seed(1234)
#models(set4fn)
```

## Set4: Nu with Completed Jobs

- Hay $4169$ trabajos completados.

```{r}
na.counts <- dim(set4 %>% filter(if_any(everything(), is.na)))[1]
set4 <- na.omit(set4)
cero.data <- colSums(set4==0)
set4 <- set4 %>% filter(CPUTimeRAW!=0)
set4 <- set4 %>% filter(State == "COMPLETED") #La mayoría son fallidos.

set4 <- cpupower.function(set4, "ConsumedEnergyRaw", "CPUTimeRAW") 
set4$TimelimitRaw <- as.numeric(set4$TimelimitRaw)
set4$TimelimitRaw <- set4$TimelimitRaw*60
set4$ReqMem <- sapply(set4$ReqMem, memory.function)

set4$Submit <- ymd_hms(set4$Submit)
set4$SubmitHour <- hour(set4$Submit)
set4$SubmitWeekday <- wday(set4$Submit)

set4$QOS <- as.factor(set4$QOS)
set4$Partition <- as.factor(set4$Partition)
set4$SubmitHour <- as.factor(set4$SubmitHour)
set4$SubmitWeekday <- as.factor(set4$SubmitWeekday)
set4 <- set4 %>% select(CPUPower, everything())

cpu.time.reserved.nonzero <- set4 %>% filter(set4$ResvCPURAW!=0)
set4$ResvCPURAW[set4$ResvCPURAW==0] <- min(cpu.time.reserved.nonzero$ResvCPURAW)

set4 <- select(set4, -ConsumedEnergyRaw, -CPUTimeRAW, -State, -Submit)

vars_to_winsor <- c("CPUPower", "ReqMem", "Priority", "TimelimitRaw","ResvCPURAW")
for (var in vars_to_winsor) {
  set4[[var]] <- winsorize(set4[[var]], threshold = 0.15)
}

#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
set4$CPUs.per.node<- set4$ReqCPUS / set4$ReqNodes
set4$Mem.per.CPU <- set4$ReqMem / set4$ReqCPUS
set4$CPU.time.requested <- set4$ReqCPUS * set4$TimelimitRaw

set4$CPUPower <- log(set4$CPUPower)
set4$CPUs.per.node <-  log(set4$CPUs.per.node)
set4$Mem.per.CPU <- log(set4$Mem.per.CPU)
set4$CPU.time.requested <- log(set4$CPU.time.requested)
set4$ResvCPURAW <- log(set4$ResvCPURAW)

set4 <- set4 %>% select(-c("ReqCPUS", "ReqNodes", "TimelimitRaw", "ReqMem", "Priority"))
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW



set4f <- set4 %>% select_if(is.factor)
set4n <- set4 %>% select_if(is.numeric) 
set4n <- as.data.frame(scale(set4n, center=T, scale=T)) 
set4n <- set4n %>% select(CPUPower, everything())
set4fnc <- cbind(set4f,set4n)

ggpairs(set4n)
corr(set4n)

set.seed(1234)
models(set4fnc)
```

# Calibración de parámetros

## Andalan - KNN y RF

```{r eval = FALSE}
calibration.knn(set1fnc)
```

knn optimal

```{r eval = FALSE}
#set1.rf <- randomForest(CPUPower ~., data = set1fn, ntree=200, keep.forest = FALSE, importance = FALSE)
#plot(set1.rf, log = "y", title = "")
```

## Dribe - KNN y RF

```{r eval = FALSE}
calibration.knn(set2fnc)
```

knn kernel inv

```{r eval = FALSE}
#set2.rf <- randomForest(CPUPower ~., data = set2fn, ntree=200, keep.forest = FALSE, importance = FALSE)
#plot(set2.rf, log = "y", title = "")
```

## Kurá - KNN y RF

```{r eval = FALSE}
calibration.knn(set3fnc)
```

```{r eval = FALSE}
#set3.rf <- randomForest(CPUPower ~., data = set3fn, ntree=200, keep.forest = FALSE, importance = FALSE)
#plot(set3.rf, log = "y", title = "")
```

## Nu - KNN y RF

```{r eval = FALSE}
calibration.knn(set4fnc)
```

knn kernel optimal

```{r eval = FALSE}
#set4.rf <- randomForest(CPUPower ~., data = set4fn, ntree=200, keep.forest = FALSE, importance = FALSE)
#plot(set4.rf, log = "y", title = "")
```





