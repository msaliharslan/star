#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:14:44 2020

@author: salih
"""

import rosbag, csv
import os
import shutil

listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]	#get list of only bag files in current dir.
numberOfFiles = str(len(listOfBagFiles))

count = 0
for bagFile in listOfBagFiles:
	count += 1
	print ("reading file " + str(count) + " of  " + numberOfFiles + ": " + bagFile)
	#access bag
	bag = rosbag.Bag(bagFile)
	bagContents = bag.read_messages()
	bagName = bag.filename


	#create a new directory
	folder = bagName.rstrip('.bag')
	try:	#else already exists
		os.makedirs(folder)
	except:
		print ("could not create folder")
		pass
	shutil.copyfile(bagName, folder + '/' + bagName)


	#get list of topics from the bag
	listOfTopics = []
	for topic, msg, t in bagContents:
		if topic not in listOfTopics:
			listOfTopics.append(topic)


	for topicName in listOfTopics:
		#Create a new CSV file for each topic
		filename = folder + '/' + topicName.replace( '/', '_') + '.csv'
		with open(filename, 'w+') as csvfile:
			filewriter = csv.writer(csvfile, delimiter = ',')
			firstIteration = True	#allows header row
			for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
				#parse data from this instant, which is of the form of multiple lines of "Name: value\n"
				#	- put it in the form of a list of 2-element lists
				msgString = str(msg)
				msgList = msgString.split('\n')
				instantaneousListOfData = []
				for nameValuePair in msgList:
					splitPair = nameValuePair.split(':')
					for i in range(len(splitPair)):	#should be 0 to 1
						splitPair[i] = splitPair[i].strip()
					instantaneousListOfData.append(splitPair)
				#write the first row from the first element of each pair
				if firstIteration:	# header
					headers = ["rosbagTimestamp"]	#first column header
					for pair in instantaneousListOfData:
						headers.append(pair[0])
					filewriter.writerow(headers)
					firstIteration = False
				# write the value from each pair to the file
				values = [str(t)]	#first column will have rosbag timestamp
				for pair in instantaneousListOfData:
					if len(pair) > 1:
						values.append(pair[1])
				filewriter.writerow(values)
	bag.close()
print ("Done reading all " + numberOfFiles + " bag files.")