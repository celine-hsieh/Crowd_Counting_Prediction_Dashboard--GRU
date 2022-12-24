library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(tidyverse)
library(lubridate)
library(zoo)
library(data.table)
library(leaflet)
library(scales)
library(plotly)
library(reticulate)
library(rgdal)
library(devtools)
#devtools::install_github("hrbrmstr/streamgraph")
#library(streamgraph)
py_config()
options(stringsAsFactors = FALSE)

source("D:/crowd-dashboard/R/data_prep.R")

source("D:/crowd-dashboard/R/about_text.R")


### UI ###

# get a vector of issue tags
#issue_tags <- sort(unique(unlist(str_split(ccc$issues, "; "))))


header <- dashboardHeader(title = "Crowd Counting & Prediction Dashboard", titleWidth = 400)

sidebar <- dashboardSidebar(

  width = 400,

  sidebarMenu(
    menuItem("Map View", tabName = "map", icon = icon("map")),
    menuItem("Summary Plots", tabName = "plots", icon = icon("dashboard")),
    menuItem("About", tabName = "about", icon = icon("address-card"))
  ),

  fluidRow(column(width = 12,

    br(),

    br(),

    selectInput("time_selector", "Show crowd by...",
                width = 375,
                choices = c("Historical Data", "Prediction"),
                selected = "Historical Data"),

    uiOutput("time_controls"),

    pickerInput(inputId = "issues",
                label = "Filter by variables", 
                width = 350,
                choices = c("停車", "溫度", "票證資料", "雨量"),
                selected = c("停車", "溫度", "票證資料", "雨量"),
                options = list(`actions-box` = TRUE, size = 5), 
                multiple = TRUE),
    
    awesomeCheckboxGroup(inputId = "valence",
                         label = "Filter by date",
                         #choices = list("weekdays" = 1, "weekend" = 2, "holiday" = 3, "summer vacation" = 4, "festival" = 0),
                         choices = list("weekdays" = 0, "weekend" = 1),
                         selected = c(0,1),
                         inline = FALSE),

    awesomeCheckboxGroup(inputId = "violence",
                         label = "Only show place...", 
                         choices = list("國華海安商圈" = '國華海安商圈',
                                        "孔廟文化園區" = '孔廟文化園區',
                                        "安平老街" = '安平老街',
                                        "港濱軸帶" = '港濱軸帶',
                                        "赤崁園區" = '赤崁園區'),
                         selected =  "國華海安商圈",
                         inline = TRUE)
  ))
  
)

body <- dashboardBody(

  tabItems(

    tabItem(tabName = "map",

      fluidRow(

        leafletOutput("ccc_map", width = "100%", height = "650px")

      )

      # fluidRow(

      #   plotOutput("daily_event_tally_ribbon", width = "100%", height = "100px")

      # )

    ),

    tabItem(tabName = "plots",

      fluidRow(
        column(width = 4,
          box(height = 200, background = "black", width = "100%", valueBoxOutput("event_tally"))
        ),
        column(width = 8,
          box(height = 200, background = "black", title = "Historical Crowd Data", width = "100%",
            #plotlyOutput("daily_event_tally_plot", height = "125px")
            imageOutput("daily_event_tally_plot", height = "125px")
          )
        )
      ),

      fluidRow(
        column(width = 4,
          box(height = 200, background = "green", width = "100%", valueBoxOutput("participant_tally"))
        ),
        column(width = 8,
          box(height = 200, background = "green", title = "Predict crowd data", width = "100%",
            imageOutput("daily_participant_tally_plot", height = "125px")
          )
        )
      ),     
      fluidRow(
        column(width = 4,
          box(height = 200, background = "black", width = "100%", valueBoxOutput("test_tally"))
        ),
        column(width = 8,
          box(height = 200, background = "black", title = "Predict Loss & Validation", width = "100%",
            imageOutput("test_tally_plot", height = "125px")
          )
        )
      )
      # fluidRow(
      #   column(width = 4),
      #   column(width = 8,
      #     box(height = 300, background = "aqua", title = "Monthly tallies of events by issues (duplication allowed)", width = "100%",
      #       streamgraphOutput("issue_streamgraph", height = "250px", width = "100%")
      #     )
      #   )
      # )

    ),

    tabItem(tabName = "about",

      formatted_about_text

    )

  )

)

ui <- dashboardPage(header, sidebar, body, skin = "black")


## SERVER ##

server <- function(input, output, session) {
  my_data <- reactiveValues()
  # generate conditional set of widgets for selecting time frame
  output$time_controls <- renderUI({

    switch(input$time_selector,

      "Historical Data" = dateRangeInput("daterange", label = "Choose a date range",
                                    width = 375,
                                    #start = Sys.Date() - 366, end = Sys.Date() - 1,
                                    #min = "2019-06-01", max = Sys.Date() - 1,
                                    start = "2022-07-01", end = "2022-08-31",
                                    min = "2022-07-01", max = "2022-08-31",
                                    #format = "yyyy-MM-D"),
                                    format = "MM, d, yyyy"),

      "Prediction" = selectInput("hour", label = "After hour",
                           width = 375,
                           #choices = rev(seq(6,9)),
                           choices = list("after 1hr" = 1, "after 2hr" = 2, "after 3hr" = 3, "after 4hr" = 4, "after 5hr" = 5),
                           selected = "after 1hr")

    )

  })

  # generate filtered and tallied data based on user selections
  ccc_munged_list <- reactive({

    dat <- ccc

    # filter for political valence
    dat <- filter(dat, valence %in% as.numeric(input$valence))

    # filter for user-selected forms of violence
    for(j in input$violence) {

      dat <- dat[dat[,j] == 1,]

    }


    # filter for issue tags, defaulting to 'other/unknown' if none is selected
    if(length(input$issues) > 0) {

      dat <- as.data.table(dat)[issues %like% paste(input$issues, collapse = "|")]

    } else {

      dat <- as.data.table(dat)[issues %like% "other/unknown"]

    }

    # now filter for time frame, conditioning on status of input$time_selector
    if(input$time_selector == "Prediction") {

      req(input$hour)

      dat <- filter(dat, lubridate::hour(date) == input$hour)

      # if(input$hour == 2 || input$hour == 4){
      #   grid <- data.frame(date = seq(date(sprintf("2022-%s-01", input$hour)), date(sprintf("2022-%s-30", input$hour)), by = "day"))
      # }else {
      #    grid <- data.frame(date = seq(date(sprintf("2022-%s-01", input$hour)), date(sprintf("2022-%s-31", input$hour)), by = "day"))
      # }

      if(input$hour == 2 || input$hour == 4){
        grid <- data.frame(date = seq(date(sprintf("2022-07-01")), date(sprintf("2022-07-31")), by = "day"))
      }else {
         grid <- data.frame(date = seq(date(sprintf("2022-08-01")), date(sprintf("2022-08-31")), by = "day"))
      }
      

      tally_events <- dat %>%
        count(date) %>%
        left_join(grid, .) %>%
        mutate(count = ifelse(is.na(n), 0, n))

      tally_participants <- dat %>%
        group_by(date) %>%
        summarize(n = sum(size_mean, na.rm = TRUE)) %>%
        ungroup() %>%
        left_join(grid, .) %>%
        mutate(count = ifelse(is.na(n), 0, n))

    } else {

      req(input$daterange)

      dat <- filter(dat, date >= input$daterange[1] & date <= input$daterange[2])

      grid <- data.frame(date = seq(date(input$daterange[1]), date(input$daterange[2]), by = "day"))

      tally_events <- dat %>%
        count(date) %>%
        left_join(grid, .) %>%
        mutate(count = ifelse(is.na(n), 0, n))

      tally_participants <- dat %>%
        group_by(date) %>%
        summarize(n = sum(size_mean, na.rm = TRUE)) %>%
        ungroup() %>%
        left_join(grid, .) %>%
        mutate(count = ifelse(is.na(n), 0, n))
      
    }

    dat_list <- list("events" = dat, "event tally" = tally_events, "participant tally" = tally_participants)

    return(dat_list)

  })

  output$ccc_map <- renderLeaflet({

    req(input$issues)

    addLegendCustom <- function(map, colors, labels, sizes, opacity = 1, position = "topright", title = NULL){

      colorAdditions <- paste0(colors, "; border-radius: 50%; width:", sizes * 2, "px; height:", sizes * 2, "px")
      labelAdditions <- paste0("<div style='display: inline-block;height: ", sizes, "px;margin-top: 4px;line-height: ", sizes, "px;'>", labels, "</div>")
    
      return(addLegend(map, title = title, colors = colorAdditions, labels = labelAdditions, opacity = opacity, position = position))

    }


    #library("rgdal")
    #shapeData <- readOGR(".",'myGIS')
    #mapfile <- normalizePath(file.path("D:/crowd-dashboard/等時圈",
                                #paste(input$violence, "/", input$violence, ".shp", sep="")))
    #shapeData <- readOGR(mapfile, layer = input$violence, GDAL1_integer64_policy = TRUE)
    shapeData1 <- readOGR("D:/crowd-dashboard/等時圈/億載金城等時圈/億載金城等時圈1.shp", layer = "億載金城等時圈1", GDAL1_integer64_policy = TRUE)
    shapeData1_2 <- readOGR("D:/crowd-dashboard/等時圈/億載金城等時圈/億載金城等時圈2.shp", layer = "億載金城等時圈2", GDAL1_integer64_policy = TRUE)

    shapeData <- readOGR("D:/crowd-dashboard/等時圈/孔廟文化園區/孔廟文化園區.shp", layer = "孔廟文化園區", GDAL1_integer64_policy = TRUE)
    shapeData_2 <- readOGR("D:/crowd-dashboard/等時圈/孔廟文化園區/孔廟等時圈2.shp", layer = "孔廟等時圈2", GDAL1_integer64_policy = TRUE)

    shapeData3 <- readOGR("D:/crowd-dashboard/等時圈/安平古堡等時圈/安平古堡等時圈1.shp", layer = "安平古堡等時圈1", GDAL1_integer64_policy = TRUE)
    shapeData3_2 <- readOGR("D:/crowd-dashboard/等時圈/安平古堡等時圈/安平古堡等時圈2.shp", layer = "安平古堡等時圈2", GDAL1_integer64_policy = TRUE)

    shapeData4 <- readOGR("D:/crowd-dashboard/等時圈/安平樹屋等時圈/安平樹屋等時圈1.shp", layer = "安平樹屋等時圈1", GDAL1_integer64_policy = TRUE)
    shapeData4_2 <- readOGR("D:/crowd-dashboard/等時圈/安平樹屋等時圈/安平樹屋等時圈2.shp", layer = "安平樹屋等時圈2", GDAL1_integer64_policy = TRUE)

    shapeData2 <- readOGR("D:/crowd-dashboard/等時圈/漁光島等時圈/漁光島等時圈1.shp", layer = "漁光島等時圈1", GDAL1_integer64_policy = TRUE)
    shapeData2_2 <- readOGR("D:/crowd-dashboard/等時圈/漁光島等時圈/漁光島等時圈2.shp", layer = "漁光島等時圈2", GDAL1_integer64_policy = TRUE)

    shapeData <- spTransform(shapeData, CRS("+proj=longlat +ellps=WGS84"))

    leaflet() %>% addTiles() %>% 
#     addProviderTiles("CartoDB.Positron") %>%
      addProviderTiles("Stamen.TonerLite") %>%
      # set view to geographic center of the united states
      setView(lat = 23.0, lng = 120.20, zoom = 13) %>%
      # addCircleMarkers(data = ccc_munged_list()[['events']],
      #                  lat = ~lat, lng = ~lon,
      #                  radius = ~marker_radius, stroke = FALSE, fillColor = ~marker_color, opacity = 1/2, group = "circles",
      #                  popup = ~marker_label) %>%
      addLegendCustom(#title = "Crowd size",
                      title = "等時圈",
                      #colors = c("gray", rep("#cc5500", 4), rep("#ffdd00", 3)),
                      colors = c("gray", rep("#ffdd00", 3)),
                      #labels = c("unknown", "10s", "100s", "1,000s", "10,000s", "等時圈 5 min", "等時圈 10 min","等時圈 15 min"),
                      labels = c("unknown", "5 min", "10 min", "15 min"),
                      #sizes = c(5,5,10,15,20,10,15,20),
                      sizes = c(5,10,15,20),
                      opacity = 1/2) %>%
      #addPolygons(data = shapeData, weight=5, col = 'red') %>% 
      #addMarkers(lng = -106.363590, lat=31.968483, popup = "Hi there")

    #leaflet()  %>% addTiles() %>% 
    #setView(lng = -106.363590, lat=31.968483,zoom=11) %>% 
      addPolygons(data=shapeData1, weight=2, col = '#ffdd00') %>% 
      addCircleMarkers(data=shapeData1_2,
                        radius = 3,
                        color ='#573600',
                        stroke = FALSE, fillOpacity = 0.5
      ) %>% 

      addPolygons(data=shapeData, weight=2, col = '#ffdd00') %>% 
      addCircleMarkers(data=shapeData_2,
                        radius = 3,
                        color ='#573600',
                        stroke = FALSE, fillOpacity = 0.5
      ) %>% 

      addPolygons(data=shapeData3, weight=2, col = '#ffdd00') %>% 
      addCircleMarkers(data=shapeData3_2,
                        radius = 3,
                        color ='#573600',
                        stroke = FALSE, fillOpacity = 0.5
      ) %>% 
      
      addPolygons(data=shapeData4, weight=2, col = '#ffdd00') %>% 
      addCircleMarkers(data=shapeData4_2,
                        radius = 3,
                        color ='#573600',
                        stroke = FALSE, fillOpacity = 0.5
      ) %>% 

      addPolygons(data=shapeData2, weight=2, col = '#ffdd00') %>% 
      addCircleMarkers(data=shapeData2_2,
                        radius = 3,
                        color ='#573600',
                        stroke = FALSE, fillOpacity = 0.5
      )
      #addPolygons(data=shapeData2_2, weight=2, col = '#573600')
      #addPolygons(data=shapeData3, weight=2, col = '#ffdd00') %>% 
      #addMarkers(data=shapeData,lng = -106.363590,lat=31.968483,popup="Hi there") 
      #addMarkers(data=shapeData, popup = input$violence)

  })

  # output$daily_event_tally_ribbon <- renderPlot({

  #   req(input$issues)

  #   ccc_munged_list()[['event tally']] %>%
  #     ggplot(aes(date, count)) +
  #       geom_col(fill = "gray60") +
  #       theme_minimal() +
  #       scale_y_continuous(position = "left", labels = comma) +
  #       scale_x_date(date_labels = "%b %d, %Y") +
  #       labs(caption = "daily event count") +
  #       theme(axis.title.x = element_blank(),
  #             axis.title.y = element_blank(),
  #             plot.caption = element_text(face = "bold", hjust = 0.5, size = 12))

  # })

  #output$daily_event_tally_plot <- renderPlotly({
  output$daily_event_tally_plot <- renderImage({

    # req(input$issues)

    # p <- ccc_munged_list()[['event tally']] %>%
    #   ggplot(aes(date, count)) +
    #     geom_col() +
    #     theme_minimal() +
    #     scale_y_continuous(position = "left", labels = comma) +
    #     scale_x_date(date_labels = "%b %d, %Y") +
    #     theme(title = element_blank(),
    #           axis.title.x = element_blank(),
    #           axis.title.y = element_blank())

    # ggplotly(p)
    if (is.null(input$violence)) {
      print("selected a place")
      filename <- normalizePath(file.path("D:/crowd-dashboard/歷史資料new",
                                paste("selected", ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    } else if(length(input$violence)>1){
       print("selected only one place")
      filename <- normalizePath(file.path("D:/crowd-dashboard/歷史資料new",
                                paste("onlyone", ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    } else {   
      use_condaenv("base")
      source_python("D:/crowd-dashboard/python/population.history.py")
      #print(input$violence)
      #print(input$daterange[1])
      #print(input$daterange[2])
      #input1 <- input$daterange[1]
      #input2 <- input$daterange[2]
      #average <- history(input$violence, input1, input2)
      my_data$average <- history(input$violence, input$daterange[1], input$daterange[2])
      #my_data <- reactiveValues(average = history(input$violence, input$daterange[1],input$daterange[2]))

      filename <- normalizePath(file.path("D:/crowd-dashboard/歷史資料new",
                                paste(input$violence, ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    }
  }, deleteFile = FALSE)
  

  output$daily_participant_tally_plot <- renderImage({
    if (is.null(input$violence)) {
      print("selected a place")
      filename <- normalizePath(file.path("D:/crowd-dashboard/預測資料new",
                                paste("selected", ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    } else if(length(input$violence)>1){
       print("selected only one place")
      filename <- normalizePath(file.path("D:/crowd-dashboard/預測資料new",
                                paste("onlyone", ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    } else {   
      use_condaenv("base")
      source_python("D:/crowd-dashboard/python/population.predict.py")
      my_data$average2 <- predict(input$violence, input$hour[1])
      print(input$violence)
      filename <- normalizePath(file.path("D:/crowd-dashboard/預測資料new",
                                paste(input$violence, ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    }
  }, deleteFile = FALSE)

  # Render the matplotlib plot as an image
  use_condaenv("base")
  output$test_tally_plot <- renderImage({

    #if (input$violence == "港濱軸帶") {
    if (is.null(input$violence)) {
      print("selected a place")
      filename <- normalizePath(file.path("D:/crowd-dashboard/人流系統Loss圖",
                                paste("selected", ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    } else if(length(input$violence)>1){
       print("selected only one place")
      filename <- normalizePath(file.path("D:/crowd-dashboard/人流系統Loss圖",
                                paste("onlyone", ".png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    } else {   
      #use_condaenv("base")
      #source_python("D:/crowd-dashboard/python/population.py")
      #map_harvest(input$violence)
      print(input$violence)
      filename <- normalizePath(file.path("D:/crowd-dashboard/人流系統Loss圖",
                                paste(input$violence, "/", input$violence, "_", "全.png", sep="")))
      return(list(
        src = filename,
        filetype = "image/png",
        alt = input$violence,
        width="100%",
        height="100%"
      ))
    }
  }, deleteFile = FALSE)


output$event_tally <- renderValueBox({

    #req(input$issues)

    #box_sum = sum(ccc_munged_list()[['event tally']][,'count'], na.rm = TRUE)
  if (is.null(input$violence)) {
      print("selected a place")
      valueBox(value = "",
               subtitle = "",
               color = "black")
    } else if(length(input$violence)>1){
      print("selected only one place")
      valueBox(value = "",
               subtitle = "",
               color = "black")

    }else {
       valueBox(value = format(my_data$average, big.mark = ","),
             subtitle = "Average",
             color = "black")
    }
    

  })


  output$participant_tally <- renderValueBox({

   if (is.null(input$violence)) {
      print("selected a place")
      valueBox(value = "",
               subtitle = "",
               color = "green")
    } else if(length(input$violence)>1){
      print("selected only one place")
      valueBox(value = "",
               subtitle = "",
               color = "green")

    }else {
       valueBox(
             value = format(my_data$average2, big.mark = ","),
             #value = "",
             subtitle = "Average",
             color = "green")
    }

  })

  output$test_tally <- renderValueBox({

    #req(input$issues)

    #box_sum = sum(ccc_munged_list()[['participant tally']][,'count'], na.rm = TRUE)

    valueBox(value = "",
             subtitle = "",
             color = "black")

  })
}


## RUN ###

shinyApp(ui = ui, server = server)