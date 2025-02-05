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
l_s = 15
l_t = 1.2
a_s = 20
a_t = 10
p_t = 20
min_max_transform <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
df1 <- read.csv("datasets/interpolation-T80.csv", , header = T)
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

df1 <- read.csv("datasets/interpolation-T90.csv", , header = T)
df1$LATITUDE <- min_max_transform(df1$LATITUDE)
df1$LONGITUDE <- min_max_transform(df1$LONGITUDE)
p4 <- ggplot(data = df1, 
         aes(x = LONGITUDE, 
             y = LATITUDE, 
             fill = median))+
    geom_tile() + scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-1,6),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() +  ggtitle(paste("Predicted median"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

p5 <- ggplot(data = df1, 
         aes(x = LONGITUDE, 
             y = LATITUDE, 
             fill = ub))+
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

p6 <- ggplot(data = df1, 
         aes(x = LONGITUDE, 
             y = LATITUDE, 
             fill = median))+
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

df1 <- read.csv("datasets/interpolation-T95.csv", , header = T)
df1$LATITUDE <- min_max_transform(df1$LATITUDE)
df1$LONGITUDE <- min_max_transform(df1$LONGITUDE)
head(df1)
p7 <- ggplot(data = df1, 
         aes(x = LONGITUDE, 
             y = LATITUDE, 
             fill = median)) +
    geom_tile() + scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-1,6),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() +  ggtitle(paste("Predicted median"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

p8 <- ggplot(data = df1, 
         aes(x = LONGITUDE, 
             y = LATITUDE, 
             fill = ub)) +
    geom_tile() + scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-1,6),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() +  ggtitle(paste("Predicted ub"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

p9 <- ggplot(data = df1, 
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
print(dim(df1))
rm(df1)
df = read.csv("datasets/dataset-10DAvg.csv", header = T)
head(df)
df_filtered <- subset(df, time_scaled == 0.800498753117207)
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
    coord_fixed() + ggtitle(paste("15 October 2022"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

df_filtered <- subset(df, time_scaled == 0.9002493765586035)
df_filtered <- df_filtered[complete.cases(df_filtered[, c("LONGITUDE", "LATITUDE", "Station_Value")]), ]
p11 <- ggplot(df_filtered,aes(x = LONGITUDE, 
             y = LATITUDE, 
             color = Station_Value)) +
    geom_point(size = 0.8, alpha = 0.4) + scale_color_gradientn(colours = nasa_palette,
                        limits = c(-1, 6),
                        name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("18 September 2023"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

df_filtered <- subset(df, time_scaled == 0.9501246882793017)
df_filtered <- df_filtered[complete.cases(df_filtered[, c("LONGITUDE", "LATITUDE", "Station_Value")]), ]
head(df_filtered)
print(dim(df_filtered))
p12 <- ggplot(df_filtered,aes(x = LONGITUDE, 
             y = LATITUDE, 
             color = Station_Value)) +
    geom_point(size = 0.8, alpha = 0.4) + scale_color_gradientn(colours = nasa_palette,
                        limits = c(-1, 6),
                        name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("07 May 2024"))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

g <- grid.arrange(p10,p11,p12,
        p1,p4,p7,
        p2,p5,p8,
        p3,p6,p9,nrow = 4)
print("I am here 2!")
ggsave(g, file = paste0("anim/interpolation.png"),
         height = 12.5, width = 12.5)

