import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    p_sleep_ft = [2.232363, 1.861881, 1.677609, 1.59006, 1.527819, 1.484307, 1.447551, 1.392255
        , 1.324506, 1.325505]
    # Data for the second stack
    p_rest_ft = [12.8194280369921, 15.3158567551066, 16.8846152027719, 17.9896488418453
        , 19.012128144407, 19.9855861705758, 20.9769934569929, 22.1058027602618, 23.351979441355, 24.3270269654707]

    p_sleep_tt = [2.738424,2.521359,2.398386,2.318484,2.270928,2.224707,2.180832,2.153385,2.114748,2.087304]

    p_rest_tt = [10.2779093262401,12.010529426812,13.2872710151134,14.372719066836,15.3118762095682
        ,16.2798779882973, 17.2617012137972,18.2117808416348,19.2604013354919,20.3079891611588]

    print(len(p_sleep_ft), len(p_rest_ft), len(p_sleep_tt), len(p_rest_tt))


    # Creating x-axis labels
    x = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    print(len(x))

    # Creating the stacked bar chart
    # plt.bar(x, data1, label='Data 1', color='b')
    plt.bar(x, p_rest_tt, color='b')
    plt.bar(x, p_rest_tt, label='Data 2', color='g', bottom=p_sleep_ft)

    # Adding labels and title
    plt.xlabel('X')
    plt.ylabel('Value')
    plt.title('Stacked Bar Chart')

    # Showing the legend
    plt.legend()

    # Displaying the chart
    plt.show()


