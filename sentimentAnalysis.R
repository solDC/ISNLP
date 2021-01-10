library(dplyr)
library(NLP) #dependencia de tm
library(tm)
library(SnowballC)
library(ggplot2)
library(wordcloud)
library(RColorBrewer) #dependencia de wordcloud
library(RWeka)
library(rpart)
library(rpart.plot)
library(e1071)
library(nnet)
library(fastTextR)

#setear el directorio raiz a donde se encuentra la "raíz" del repo (source file location)
setwd("~/Documents/ISNLP/")

set.seed(123) 

#cargo dataset
reviews_big <- read.csv("./datos/train.csv", stringsAsFactors = F, header = F)

#me quedo con muestra aleatoria que la muestra es muy grande
reviews <- reviews_big[sample(nrow(reviews_big),5000),]
str(reviews)
#convierto la clase en un factor y le asigno etiquetas
reviews$V1 <- factor(reviews$V1)
levels(reviews$V1) <- c("Neg", "Pos")
table(reviews$V1)

#me quedo con reseñas pos y neg por separado
#reviews_pos <- reviews %>% filter(V1 == 2) %>% select(V3)
#reviews_neg <- reviews %>% filter(V1 == 1) %>% select(V3)

#transformo los textos (cada review sería un documento) en un corpus con el paquete tm
reviews_src <- VectorSource(reviews$V3)
reviews_corpus <- VCorpus(reviews_src)

#inspecciono el corpus
print(reviews_corpus)
inspect(reviews_corpus[1:5]) 
as.character(reviews_corpus[[1]]) #lapply(reviews_corpus[1:5], as.character)
reviews_corpus[[1]]$content[1]

#Para analizar el texto, hay que dividir el texto en palabras, pero primero hay que limpiar y estandarizar el texto 
reviews_corpus <- tm_map(reviews_corpus,
                         content_transformer(tolower))#para usar solo minusculas
lapply(reviews_corpus[1:5], as.character)

#getTransformations()
reviews_corpus <- tm_map(reviews_corpus,removePunctuation) #elimino signos de puntuación
reviews_corpus <- tm_map(reviews_corpus,removeNumbers) #elimino números
#stopwords() #para ver cuales son las stopwords
#reviews_pos_corpus <- tm_map(reviews_corpus_clean,removeWords,stopwords())
#agrego más stopwords que se me ocurren qeu si aparecen en una reseña no aportan valor, habrá que inspeccionar si aparecen más
myStopwords = c(stopwords(),"review","reviews","ve","s","the") #purchase sería otra buena palabra
reviews_corpus <- tm_map(reviews_corpus,removeWords,myStopwords) # elimino stopwords
reviews_corpus <- tm_map(reviews_corpus,stemDocument) #me quedo con la raiz de las palabras

reviews_corpus <- tm_map(reviews_corpus,stripWhitespace) #elimino espacios extras, tiene que hacerse en este orden
lapply(reviews_corpus[1:5], as.character)

#sigo con la preparación de datos. Voy a dividiar el texto en palabra
#Uso una matriz documento-termino TDM donde los documentos están en las columnas y los terminos como filas Peso por efault: frecuencia
reviews_tdm <- TermDocumentMatrix(reviews_corpus)
reviews_tdm

findFreqTerms(reviews_tdm,100)

#me daba error de memoria con muchos registros error de memoria
reviews_tdm_matrix <- as.matrix(reviews_tdm)
reviews_tdm_matrix[500:600,1:20]

freq <- rowSums(as.matrix(reviews_tdm))
high.freq=tail(sort(freq),n=15)
hfp.df=as.data.frame(sort(high.freq))
hfp.df$names <- rownames(hfp.df) 
ggplot(hfp.df, aes(reorder(names,high.freq), high.freq)) +
  geom_bar(stat="identity", fill="darkred") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")

wordcloud(reviews_corpus,min.freq = 60,random.order = FALSE)

#Use Weka’s n-gram tokenizer to create a TDM that uses as terms the bigrams that appear in the corpus.Usa library tau
options(mc.cores=1)
BigramTokenizer <- function(x) {RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 2, max = 2))}

tdm_bigram = TermDocumentMatrix(reviews_corpus,
                                control = list(tokenize = BigramTokenizer))
freq = sort(rowSums(as.matrix(tdm_bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)

pal=brewer.pal(8,"Reds")
pal=pal[-(1:3)]
png("wordcloud.png", width=800,height=800)
par(mar = rep(0, 4))
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)
dev.off()

#para disminuir el sparse
reviews_tdm_small <- removeSparseTerms(reviews_tdm,0.95)
findFreqTerms(reviews_tdm_small,100)

#genero la matriz documento término pero con uso tf-idf para medir la importancia relativa en el documento
#remuevo los sparse terms, podríamos ver las nubes de palabras otra vez para ver si son más informativas que antes
#sobre todo serviría para la de una sola palabra. Mirando los resultadis no me gusta.
review_dtm_tfidf <- DocumentTermMatrix(reviews_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf

freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
png("wordcloud-tfidf.png", width=800,height=800)
par(mar = rep(0, 4))
wordcloud(rownames(freq), max.words=100, random.order = F, colors=pal)
dev.off()

ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity", fill="darkred") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent bigrams")


# Modelo predictivo

# Voy a usar una lista precompilada con palabras positivas y negativas
neg_words <- read.table("./datos/negative-words.txt", header = F, stringsAsFactors = F, fill=T)[, 1]
pos_words <- read.table("./datos/positive-words.txt", header = F, stringsAsFactors = F, fill=T)[, 1]
#como indicador simple, creo dos variables qeu contienen el número de palabfas positivas y negativas en cada doc
reviews$neg <- sapply(reviews_corpus, tm_term_score, neg_words)
reviews$pos <- sapply(reviews_corpus, tm_term_score, pos_words)

#preparo los datos para el modelo estadístico
reviews$V2 <- NULL
reviews$V3 <- NULL
reviews <- cbind(reviews, as.matrix(review_dtm_tfidf))
str(reviews)

#parto el dataset en train y test
id_train <- sample(nrow(reviews),nrow(reviews)*0.80)
reviews.train = reviews[id_train,]
reviews.test = reviews[-id_train,]

#voy a usar paquetes 'rpart', 'rpart.plot', 'e1071', 'nnet'
reviews.tree = rpart(V1~.,  method = "class", data = reviews.train);  
prp(reviews.tree)
reviews.glm = glm(V1~ ., family = "binomial", data =reviews.train, maxit = 100);  
reviews.svm = svm(V1~., data = reviews.train);
reviews.nnet = nnet(V1~., data=reviews.train, size=1, maxit=500)

#evaluar el modelo
pred.tree = predict(reviews.tree, reviews.test,  type="class")
table(reviews.test$V1,pred.tree,dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$V1 != pred.tree, 1, 0))

pred.glm = as.numeric(predict(reviews.glm, reviews.test, type="response") > 0.5)
table(reviews.test$V1,pred.glm,dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$V1 != pred.glm, 1, 0))

pred.svm = predict(reviews.svm, reviews.test)
table(reviews.test$V1,pred.svm,dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$V1 != pred.svm, 1, 0))

prob.nnet= predict(reviews.nnet,reviews.test)
pred.nnet = as.numeric(prob.nnet > 0.5)
table(reviews.test$V1, pred.nnet, dnn=c("Obs","Pred"))
mean(ifelse(reviews.test$V1 != pred.nnet, 1, 0))

#fastText
#train <- read.table("./datos/train.ft.txt", header = F, stringsAsFactors = F, fill=T)[, 1]
#ft_normalize(train) 

#error in train
#cntrl <- ft.control(word_vec_size = 10L, learning_rate = 0.1, max_len_ngram = 2L, 
#                    min_count = 1L, nbuckets = 10000000L, epoch = 5L, nthreads = 20L)

#model <- ft_train(input = "train", method = "supervised", control = cntrl)

#ft_save(model, "reviews")
#ft_test()
#ft_predict()