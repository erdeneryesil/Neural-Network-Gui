#ifndef DEFINES_H
#define DEFINES_H

#include<QtWidgets>

#define FORM_TRANSPARENT 0.6
#define FORM_OPAQUE 1.0

#define FORM_MAIN_CONTROL_WIDTH 450
#define FORM_MAIN_CONTROL_HEIGHT 25
#define FORM_MAIN_LIST_HEIGHT 200
#define FORM_MAIN_LABEL_HEIGHT 50
#define FORM_MAIN_GRAPHICS_VIEW_WIDTH 850
#define COLOR_LINE QColor(0,0,0)
#define COLOR_ELLIPSE_BORDER QColor(255,255,255)
#define COLOR_INPUT_CELLS QColor(255,102,102)
#define COLOR_HIDDEN_CELLS QColor(102,204,0)
#define COLOR_OUTPUT_CELLS QColor(0,127,255)
#define COLOR_BIAS_CELLS QColor(255,255,102)
#define R_CELL 50
#define COLOR_DOT QColor(0,0,0)
#define R_DOT 10
#define MAX_COUNT_CELL_IN_DIAGRAM 10
#define COUNT_SHORTENED_CELL_IN_DIAGRAM 7
#define COUNT_DOT 3


#define WIDGET_STYLESHEET_LIGHT "background-color:#cccccc;"
#define WIDGET_STYLESHEET_DARK "background-color:#aaaaaa;"
#define LOCAL_FOR_DECIMAL_SEPARATOR QLocale(QLocale::English)
#define SCROLL_AREA_HEIGHT 200
#define LIMIT_MIN -99999
#define LIMIT_MAX 99999
#define TYPE_BINARY_INDEX 0
#define TYPE_FLAG_INDEX 1
#define TYPE_NARROWED_INDEX 2
#define TYPE_NATIVE_INDEX 3
#define PROPERTY_INDEX "index"
#define COUNT_DATA_TYPES 4
#define DATA_TYPES (QString[]){"Binary","Flag","Narrowed","Native"}
#define COUNT_SUM_FUNCTIONS 6
#define SUM_FUNCTIONS (QString[]){"Weighted Total","Multiplication","Maximum","Minimum","Majority","Incremental Total"}
#define COUNT_ACT_FUNCTIONS 6
#define ACT_FUNCTIONS (QString[]){"Sigmoid","TanH","ReLU","Leaky ReLU","Swish","Softplus"}

#define COUNT_INPUT_MIN 1
#define COUNT_INPUT_MAX 999

#define COUNT_HIDDEN_LAYER_MIN 1
#define COUNT_HIDDEN_LAYER_MAX 9
#define COUNT_HIDDEN_CELL_MIN 1
#define COUNT_HIDDEN_CELL_MAX 999

#define COUNT_OUTPUT_MIN 1
#define COUNT_OUTPUT_MAX 999

#define TOLERANCE_MIN 0.0
#define TOLERANCE_MAX 1.0
#define TOLERANCE_DECIMALS 5
#define TOLERANCE_SINGLE_STEP 0.01

#define TEST_RESULT_UNCERTAIN -1

#define LAMBDA_MIN 0.0
#define LAMBDA_MAX 1.0
#define LAMBDA_DECIMALS 5
#define LAMBDA_SINGLE_STEP 0.01
#define ALFA_MIN 0.0
#define ALFA_MAX 1.0
#define ALFA_DECIMALS 5
#define ALFA_SINGLE_STEP 0.01
#define RANDOM_WEIGHT_MIN -5.0
#define RANDOM_WEIGHT_MAX 5.0
#define RANDOM_WEIGHT_DECIMALS 3
#define RANDOM_WEIGHT_SINGLE_STEP 0.001
#define LENGTH_PARAMETERS_FILE_NAME 20
#define REGEX_PARAMETERS_FILE_NAME "[A-Za-z0-9_]+"
#define REGEX_PARAMETERS_INPUTS "[0-9\\s|,;-]+"
#define COUNT_SEPARATOR 4
#define SEPARATOR_DESCRIPTIONS (QString[]){"Space ","Comma ,","Semicolon ;","Pipe |"}
#define SEPARATORS (char[]){' ',',',';','|'}

#define COUNT_SAMPLE_MIN 1
#define COUNT_SAMPLE_MAX 999999

#define ERROR_GRAPH_VIEW_WIDTH 1200
#define ERROR_GRAPH_VIEW_HEIGHT 314
#define ERROR_GRAPH_SCENE_X 0
#define ERROR_GRAPH_SCENE_Y 0

#define ERROR_GRAPH_PEN_ERROR_COLOR QColor(199,0,57)
#define ERROR_GRAPH_PEN_ERROR_WIDTH 1
#define ERROR_GRAPH_PEN_WEIGHT_COLOR QColor(104,51,255)
#define ERROR_GRAPH_PEN_WEIGHT_WIDTH 1
#define ERROR_GRAPH_PEN_TRAINING_COLOR QColor(125,125,125)
#define ERROR_GRAPH_PEN_TRAINING_WIDTH 10.0
#define ERROR_GRAPH_SPACE_BETWEEN_TRAININGS 5.0
#define ERROR_GRAPH_EW_ELLIPSE_R 3.0

#define ERROR_GRAPH_PUSHBUTTON_WIDTH 25
#define ERROR_GRAPH_CURRENT_TRAINING_X_INC 1.0
#define ERROR_GRAPH_INITIAL_ERRORACTUALMAX 0.000001
#define ERROR_GRAPH_INITIAL_WEIGHTACTUALMAX 1
#define ERROR_GRAPH_XGRAPHMIN 0
#define ERROR_GRAPH_YGRAPHMIN 10
#define ERROR_GRAPH_XMIN 0
#define ERROR_GRAPH_YMIN 0

#define PARAMETERS_FILE_NAME_LABEL_WIDTH 300
#define INPUTS_LINE_EDIT_WIDTH 200

#endif // DEFINES_H
