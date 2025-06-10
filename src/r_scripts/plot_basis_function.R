library("ggplot2")
library("reticulate")
library("gridExtra")
library("viridis")
np <- import("numpy")
setwd("/home/praktik/Desktop/Spatio-temporalDeepKriging-JRC/")
df <- read.csv("basis-plot-locs.csv",header=T)
phi <- py_to_r(np$load("basis-for-plot.npy"))
l_s = 10
l_t = 1
a_s = 20
a_t = 6
p_t = 20
plot_basis <- function(i){
df$data <- phi[,i]
  b1 <- ggplot(df) +
    geom_tile(aes(x = LONGITUDE, y = LATITUDE, fill = data)) +
    scale_fill_viridis(option = "Cividis",limits = c(0,1),
                       name = "basis val") +
    theme_bw() +
    xlab("x") +
    ylab("y") +
    coord_fixed() + ggtitle(paste(" Basis number ",i))+
    theme(legend.text=element_text(size=rel(l_t)),
          legend.title = element_text(size=l_s),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

}

for(i in 1:25){
  plot <- plot_basis(i)
  ggsave(plot, file = paste0("anim/basis-",i,".png"),
         height = 3.5, width = 3.5)
}