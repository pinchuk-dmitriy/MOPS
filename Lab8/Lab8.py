import random
import numpy as np

#implementation of excel function ВПР
def VPR(randomNumbers,probability,count_of_clients): #определяет количество прибывших клиентов
    numbersOfNewComerClients=np.zeros(len(randomNumbers))
    for i in range(len(randomNumbers)):
        for j in range(0,4,1):
            if (probability[j] / 100 > randomNumbers[i]):
                break;
            if (probability[j] / 100 <= randomNumbers[i]):
                numbersOfNewComerClients[i] = count_of_clients[j]

    return numbersOfNewComerClients

def MIN(numberOfParkingLots,numbersOfNewComerClients,numberOfClientsInTheQueue):
    if(numberOfClientsInTheQueue>0 and numbersOfNewComerClients==0): return 1
    return numberOfParkingLots if numbersOfNewComerClients>=numberOfParkingLots else numbersOfNewComerClients


count_of_clients=np.array([0,1,2,3])
probability=np.array([70,12,16,2]) #вероятность
integral_probability=np.array([0,70,82,98]) #интегральная вероятность
five_minutes_intervals=np.zeros(100)#интервал в 5 минут на одного клиента
expected_clients=0 #ожидаемые_клиенты
for i in range(len(count_of_clients)): #математическое ожидание
    expected_clients+=count_of_clients[i]*probability[i]/100
print("ожидаемое число клиентов:",(expected_clients))
for i in range(0,len(five_minutes_intervals)):
    five_minutes_intervals[i]=i
print("5-минутные интервалы,номер:",five_minutes_intervals)
randomNumbers=np.zeros(100)
for i in range(0,len(randomNumbers)):
    randomNumbers[i]=random.uniform(0,1)
print("Рандомное значение между 0 и 1",randomNumbers)
randomNumbers[0]=0
numbersOfNewComerClients=VPR(randomNumbers,integral_probability,count_of_clients) #определяет количество прибывших клиентов
print("Номер клиента:",numbersOfNewComerClients)
#////////////////////////////////////
#change to second situation!!
#values=2
#////////////////////////////////////
numberOfParkingLots=2
print("number of parking lots:",numberOfParkingLots)
numberOfClientsInTheQueue=np.zeros(100)
numberOfClientsInService=np.zeros(100)
numberOfClientsInWait=np.zeros(100)
for i in range(1,len(numberOfClientsInWait)):
    numberOfClientsInTheQueue[i]=numbersOfNewComerClients[i]+numberOfClientsInWait[i-1]
    numberOfClientsInService[i]=MIN(numberOfParkingLots,numbersOfNewComerClients[i],numberOfClientsInTheQueue[i])
    numberOfClientsInWait[i]=numberOfClientsInTheQueue[i]-numberOfClientsInService[i]
print("количество клиентов в очереди:",numberOfClientsInTheQueue)
print("количество обслуживаемых клиентов:",numberOfClientsInService)
print("количество ожидающих клиентов:",numberOfClientsInWait)
print("///////////////")
print("Количество клиентов(Max):",np.max(numbersOfNewComerClients))
print("Количество клиентов(Average):",numbersOfNewComerClients.mean())
print("Количество клиентов(Sum):",numbersOfNewComerClients.sum())
print("///////////////")
print("Количество клиентов в очереди(Max):",np.max(numberOfClientsInTheQueue))
print("Количество клиентов в очереди(Average):",numberOfClientsInTheQueue.mean())
print("Количество клиентов в очереди(Sum):",numberOfClientsInTheQueue.sum())
print("///////////////")
print("количество обслуживаемых клиентов(Max)",np.max(numberOfClientsInService))
print("количество обслуживаемых клиентов(Average):",numberOfClientsInService.mean())
print("количество обслуживаемых клиентов(Sum):",numberOfClientsInService.sum())
print("///////////////")
print("количество ожидающих клиентов(Max)",np.max(numberOfClientsInWait))
print("количество ожидающих клиентов(Average):",numberOfClientsInWait.mean())
print("количество ожидающих клиентов(Sum):",numberOfClientsInWait.sum())
