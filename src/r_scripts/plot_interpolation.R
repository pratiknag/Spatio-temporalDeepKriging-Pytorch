library("ggplot2")
library("reticulate")
library("gridExtra")
library("viridis")
setwd("/home/praktik/Desktop/Spatio-temporalDeepKriging-JRC/")
nasa_palette <- c("#03006d","#02008f","#0000b6","#0001ef","#0000f6",
                  "#0428f6","#0b53f7","#0f81f3",
                  "#18b1f5","#1ff0f7","#27fada","#3efaa3","#5dfc7b",
                  "#85fd4e","#aefc2a","#e9fc0d","#f6da0c","#f5a009",
                  "#f6780a","#f34a09","#f2210a","#f50008","#d90009",
                  "#a80109","#730005")

bwr <- colorRampPalette(c("blue","white","red"))
wr <- colorRampPalette(c("brown","blue","green"))
color_array <- bwr(100)
color_array.red <- wr(100)


l_s = 10
l_t = 1
a_s = 20
a_t = 6
p_t = 20
min_max_transform <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

plot_interpolation <- function(i){
  df1 <- read.csv(paste0("datasets/interpolation/interpolation-T",i-1,".csv"), , header = T)
  df1$LATITUDE <- min_max_transform(df1$LATITUDE)
  df1$LONGITUDE <- min_max_transform(df1$LONGITUDE)
  p1 <- ggplot(data = df1, 
               aes(x = LONGITUDE, 
                   y = LATITUDE, 
                   fill = median)) +
    geom_tile() + scale_fill_gradientn(colours = nasa_palette,
                                       limits = c(-1,6),
                                       name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Predicted median"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  p2 <- ggplot(data = df1, 
               aes(x = LONGITUDE, 
                   y = LATITUDE, 
                   fill = ub)) +
    geom_tile() + scale_fill_gradientn(colours = nasa_palette,
                                       limits = c(-1,6),
                                       name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Predicted ub"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  p3 <- ggplot(data = df1, 
               aes(x = LONGITUDE, 
                   y = LATITUDE, 
                   fill = median)) +
    geom_tile() + scale_fill_gradientn(colours = nasa_palette,
                                       limits = c(-1,6),
                                       name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Predicted lb"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  return(list(p1=p1,p2=p2,p3=p3))
}

Time <- c(0.900249376558604,
       0.902743142144638,
       0.905236907730673,
       0.907730673316708,
       0.910224438902743,
       0.912718204488778,
       0.915211970074813,
       0.917705735660848,
       0.920199501246883,
       0.922693266832918,
       0.925187032418953,
       0.927680798004988,
       0.930174563591023,
       0.932668329177057,
       0.935162094763092,
       0.937655860349127,
       0.940149625935162,
       0.942643391521197,
       0.945137157107232,
       0.947630922693267,
       0.950124688279302,
       0.952618453865337,
       0.955112219451372,
       0.957605985037407,
       0.960099750623441,
       0.962593516209476,
       0.965087281795511,
       0.967581047381546,
       0.970074812967581,
       0.972568578553616,
       0.975062344139651,
       0.977556109725686,
       0.980049875311721,
       0.982543640897756,
       0.985037406483791,
       0.987531172069825,
       0.99002493765586,
       0.992518703241895,
       0.99501246882793,
       0.997506234413965)
start_date <- as.Date("2023-11-07")

# Generate 40 dates with 10-day intervals
dates <- seq.Date(from = start_date, by = "10 days", length.out = 40)
df = read.csv("datasets/dataset-10DAvg.csv", header = T)
head(df)

plot_true <- function(i){
  df_filtered <- subset(df, round(time_scaled, 10)== round(Time[i],10))
  df_filtered <- df_filtered[complete.cases(df_filtered[, c("LONGITUDE", "LATITUDE", "Station_Value")]), ]
  row.names(df_filtered) <- NULL
  p10 <- ggplot(df_filtered,aes(x = LONGITUDE, 
                                y = LATITUDE, 
                                color = Station_Value)) +
    geom_point(size = 0.8, alpha = 0.4) + scale_color_gradientn(colours = nasa_palette,
                                                                limits = c(-1, 6),
                                                                name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste(dates[i]))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  
  return(p10)
}

for(i in 1:40){
  p <- plot_interpolation(i)
  q <- plot_true(i)
  g <- grid.arrange(q,p$p1,p$p2,p$p3,
                    nrow=1,padding = unit(0.1, "cm"))
  ggsave(g, file = paste0("anim/interpolation-",i,".png"),
         height = 3, width = 12.5)
}



