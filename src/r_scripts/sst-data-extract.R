# library("animation")
library("ggplot2")
library("reticulate")
library("gridExtra")
np <- import("numpy")
setwd("/home/praktik/Desktop/Spatio-temporalDeepKriging-JRC/")
nasa_palette <- c("#03006d","#02008f","#0000b6","#0001ef","#0000f6",
                  "#0428f6","#0b53f7","#0f81f3",
                  "#18b1f5","#1ff0f7","#27fada","#3efaa3","#5dfc7b",
                  "#85fd4e","#aefc2a","#e9fc0d","#f6da0c","#f5a009",
                  "#f6780a","#f34a09","#f2210a","#f50008","#d90009",
                  "#a80109","#730005")

r_pred.med <- py_to_r(np$load("datasets/pred_forecast-FNO-med.npy"))  # Replace with your .npy file path

r_pred.lb <- py_to_r(np$load("datasets/pred_forecast-FNO-lb.npy"))
r_pred.ub <- py_to_r(np$load("datasets/pred_forecast-FNO-ub.npy"))

r_true <- py_to_r(np$load("true_forecast.npy"))
r_true_input <- py_to_r(np$load("true_forecast-input.npy"))[,,,,1]


plot_zone_preds <- function(t){
  test_mat_in <- r_true_input[t,,,1]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g1 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("True input, T=",t))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_true_input[t,,,2]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g2 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("True input, T=",t+1))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_true_input[t,,,3]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g3 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("True input, T=",t+2))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())


  test_mat_in <- r_true[t,,,1]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g4 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("True output, T=",t+3))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_true[t,,,2]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g5 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("True output, T=",t+4))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_true_input[t,,,3]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g6 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("True output, T=",t+5))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())


  test_mat_in <- r_pred.med[t,,,1]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g7 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred median, T=",t+3))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_pred.med[t,,,2]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g8 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred median, T=",t+4))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_pred.med[t,,,3]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g9 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred_median, T=",t+5))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())


test_mat_in <- r_pred.ub[t,,,1]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g10 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred ub, T=",t+3))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_pred.ub[t,,,2]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g11 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred ub, T=",t+4))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_pred.ub[t,,,3]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g12 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred_ub, T=",t+5))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())


  test_mat_in <- r_pred.lb[t,,,1]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g13 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred lb, T=",t+3))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_pred.lb[t,,,2]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g14 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred lb, T=",t+4))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())
  test_mat_in <- r_pred.lb[t,,,3]
  data_for_plot <- as.data.frame(as.table(test_mat_in))
  g15 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
    scale_fill_gradientn(colours = nasa_palette,
                         limits = c(-2,2),
                         name = "Std.mm") +
    theme_bw() +
    xlab("Longitude (deg)") +
    ylab("Latitude (deg)") +
    coord_fixed() + ggtitle(paste("Pred_lb, T=",t+5))+
    theme(axis.text.x=element_blank(),
  axis.ticks.x=element_blank(),
  axis.text.y=element_blank(),
  axis.ticks.y=element_blank())


  grid.arrange(g1, g2, g3, g4, g5,g6,
               g7,g8,g9,g10,g11,g12,g13,g14,g15,
               nrow = 5)
}

for(i in 1:20){
  plot <- plot_zone_preds(i)
  ggsave(plot, file = paste0("anim/zone1-",i,".png"),
         height = 12.5, width = 12.5)
}



