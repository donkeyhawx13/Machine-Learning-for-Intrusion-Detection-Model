About

This is a Machine Learning Intrusion Detection model and front end created for my Final Year Project.
It uses the CICIDS-2018 Dataset for training and intial testing.

Testing

For real world testing a network was created using 2 laptops conencted over Ethernet togther, consisting of a 2010 intel core 2 duo polycaronate macbook with 4GB RAM and 256GB of storage running Parrot OS for attacking a virtual machine running on the second laptop a Framework 13 with a AMD 7840U and 32Gb of RAM and 2TB of Storage, the virtual machines running on the Framework 13 were a Metasploitable VM used to insentivise the attacker as a Honey-pot, the second virtual machine was an intrusion detection system (SNORT) used to monitor the network traffic for attacks.

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/c73e6005-a232-45ee-b0d8-6d5428f21b43" />

IP Addresses
Snort VM 192.168.0.1
Metasploitable VM 192.168.0.2
Parrot OS 192.168.0.10

Output

original Dataset test showed that the model was able to identify the different attacks and correctly label them into a pie chart for easy understanding
<img width="900" height="790" alt="image" src="https://github.com/user-attachments/assets/2109cbf4-a8c4-4fae-a2f9-fc6c9d024e38" />

unfortunatly the machine learning model was overfitted to the test data and when tested with the data captured from the network it couldnt identify any attacks on the system, this can be fixed by reajusting and retraining the machine learning algorithm with different datasets and changing varibles within the Support Vector Machine Algorithm that was used on the Dataset.
<img width="790" height="790" alt="image" src="https://github.com/user-attachments/assets/6c25e07f-85e0-4b6d-8dcc-27e580b1b911" />

At this current moment it has issues with overfitting to the training data and can not correctly identify many types of attacks on a network this can be easily fixed by retraining the model with modified varibles for better outcomes in testing

last updated 21st September 2025
