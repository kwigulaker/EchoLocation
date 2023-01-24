"""
Datagram reading logic for conversion between .kmall to .xyz, with temporary .pings file in between.
The logic is derived from the file "doItAllKmallNoSeabedImageNoTide.py".
"""
import os
import struct
clear = lambda: os.system('cls')
times = []
tides = []
noTides = -1
min_e = min_n = 99.0
max_e = max_n = 0.0
start_t = stop_t = -1
lastidt = 0
stdstr = ""

# Process one depth datagram, #MRZ
# lengtha and chunk are from processDatagram, see below
# millisec is decoded from the header, so I send it in as a parameter here
def process_MRZ_data(millisec, lengtha, chunk, outputIO):
	global min_e
	global min_n
	global max_e
	global max_n
	global start_t
	global stop_t
	global stdstr
	XY_DECIMALS = 8
	
	# Headersize is 4 bytes smaller than in the headerfile, remember that the 4
	# bytes with the length has been dropped
	headersize = 1 + 1 + 1 + 1 + 1 + 1 + 2 + 4 + 4
	partitionsize = 2 + 2
	commonsize = 2 + 2 + 8
	common = struct.Struct('HHBBBBBBBB')
	numBytesCmnPart, pingCnt, rxFansPerPing, rxFanIndex, swathsPerPing, swathAlongPosition, \
	txTransducerInd, rxTransducerInd, numRxTransducers, algorithmType = common.unpack_from(chunk, headersize + partitionsize)
	pinginfo_size = 2 + 2 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 11 * 4 + 2 + 2 + 1 + 1 + 2 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 2 + 2 + 6 * 4 + 1 + 1 + 1 + 1 + 8 + 8 + 4 + 8
	pinginfo = struct.Struct('HHfBBBBBBHfffffffffffhhBBHIfffHHfHHffffffBBBBddf')
	numBytesInfoData, padding0, pingRate_Hz, beamSpacing, depthMode,\
	subDepthMode, distanceBtwSwath, detectionMode, pulseForm, \
	padding01, frequencyMode_Hz, freqRangeLowLim_Hz, \
	freqRangeHighLim_Hz, maxTotalTxPulseLength_sec, \
	maxEffTxPulseLength_sec, maxEffTxBandWidth_Hz, \
	absCoeff_dBPerkm, portSectorEdge_deg, \
	starbSectorEdge_deg, portMeanCov_deg, \
	starbMeanCov_deg, portMeanCov_m, \
	starbMeanCov_m, modeAndStabilisation, \
	runtimeFilter1, runtimeFilter2,\
	pipeTrackingStatus, transmitArraySizeUsed_deg,\
	receiveArraySizeUsed_deg, transmitPower_dB,\
	SLrampUpTimeRemaining, padding1,\
	yawAngle_deg, numTxSectors, numBytesPerTxSector,\
	headingVessel_deg, soundSpeedAtTxDepth_mPerSec,\
	txTransducerDepth_m, z_waterLevelReRefPoint_m, \
	x_txTransducerArm_SCS_m, y_txTransducerArm_SCS_m,\
	latLongInfo, posSensorStatus, attitudeSensorStatus,\
	padding2, latitude_deg, longitude_deg,\
	ellipsoidHeightReRefPoint_m = pinginfo.unpack_from(chunk, headersize + partitionsize + commonsize)
	
	# Fix of a bug in Python, where binary alignments are not correct
	latlon = struct.Struct("d")
	klat = latlon.unpack_from(chunk, headersize + partitionsize + commonsize + 124)
	klon = latlon.unpack_from(chunk, headersize + partitionsize + commonsize + 124 + 8)
	ellheight = struct.Struct("f")
	ellipsheight = ellheight.unpack_from(chunk, headersize + partitionsize + commonsize + 124 + 8 + 8)
	latitude_deg = klat[0]
	longitude_deg = klon[0]
	ellipsoidHeightReRefPoint_m = ellipsheight[0]
	
	# Pointer offset to sectorInfo
	sectorInfo_offset = headersize + partitionsize + commonsize + pinginfo_size
	# Changed from version 0
	sectorInfo = struct.Struct('BBBBfffffffBBHfff')
	sectorInfo_size = 1 + 1 + 1 + 1 + 7 * 4 + 1 + 1 + 2 + 4 + 4 + 4
	i = 0
	while (i < numTxSectors):
		txSectorNumb, txArrNumber, txSubArray, padding0,\
		sectorTransmitDelay_sec, tiltAngleReTx_deg,\
		txNominalSourceLevel_dB, txFocusRange_m,\
		centreFreq_Hz, signalBandWidth_Hz, \
		totalSignalLength_sec, pulseShading, signalWaveForm,\
		padding1, highVoltageLevel_dB, sectorTrackingCorr_dB, effectiveSignalLength_sec = sectorInfo.unpack_from(chunk, sectorInfo_offset + i * sectorInfo_size)
		i+=1

	rxInfo_offset = sectorInfo_offset + numTxSectors * sectorInfo_size
	rxInfo = struct.Struct('HHHHffffHHHH')
	rxInfo_size = 2 + 2 + 2 + 2 + 4 + 4 + 4 + 4 + 2 + 2 + 2 + 2
	numBytesRxInfo, numSoundingsMaxMain, numSoundingsValidMain, numBytesPerSounding, \
	WCSampleRate, seabedImageSampleRate, BSnormal_dB, BSoblique_dB, \
	extraDetectionAlarmFlag, numExtraDetections, numExtraDetectionClasses, \
	numBytesPerClass = rxInfo.unpack_from(chunk, rxInfo_offset)
	extraDetClassInfo_offset = rxInfo_offset + rxInfo_size
	extraDetectionSize = 2 + 1 + 1
	extraDetectionStruct = struct.Struct('HBB')
	sounding_offset = extraDetClassInfo_offset + numExtraDetectionClasses * extraDetectionSize
	soundingStruct = struct.Struct('HBBBBBBBBHffffffHHffffffffffffffffffHHHH')
	sounding_size = 2 + 8 + 2 + 6 * 4 + 2 + 2 + 18 * 4 + 4 * 2

    # Offset to seabed image
	seabedImageStart = sounding_offset + (sounding_size * (numSoundingsMaxMain + numExtraDetections))
	seabedStruct = struct.Struct('h')
	sbed_len = lengtha + 4 - seabedImageStart - 4
	tot_no_sbed = sbed_len / 2
	verify_length = tot_no_sbed * 2
	lenStruct = struct.Struct('I')
	dgmlenver = seabedImageStart + sbed_len
	dgmlen = lenStruct.unpack_from(chunk,dgmlenver - 4)[0] # should be 4 more then lengtha

	outputstr = f"\n%.{XY_DECIMALS}f %.{XY_DECIMALS}f %.2f %.2f %d\n" % (latitude_deg, longitude_deg, 
		ellipsoidHeightReRefPoint_m, z_waterLevelReRefPoint_m, millisec)
	outputIO.write(outputstr)
	sbed_start = seabedImageStart # This is the pointer to the start of the seabed image for current beam
	no_sbed_found = 0
	i = 0
	stdstr = ""
	while(i < numSoundingsMaxMain):
		soundingIndex, txSectorNumb, detectionType, \
		detectionMethod, rejectionInfo1, rejectionInfo2, \
		postProcessingInfo, detectionClass, detectionConfidenceLevel, \
		padding, rangeFactor, qualityFactor, \
		detectionUncertaintyVer_m, detectionUncertaintyHor_m, \
		detectionWindowLength_sec, echoLength_sec, \
		WCBeamNumb, WCrange_samples, WCNomBeamAngleAcross_deg, \
		meanAbsCoeff_dBPerkm, reflectivity1_dB, reflectivity2_dB, \
		receiverSensitivityApplied_dB, sourceLevelApplied_dB, \
		BScalibration_dB, TVG_dB, beamAngleReRx_deg, \
		beamAngleCorrection_deg, twoWayTravelTime_sec, \
		twoWayTravelTimeCorrection_sec, deltaLatitude_deg, \
		deltaLongitude_deg, z_reRefPoint_m, y_reRefPoint_m, \
		x_reRefPoint_m, beamIncAngleAdj_deg, realTimeCleanInfo, \
		SIstartRange_samples, SIcentreSample, \
		SInumSamples = soundingStruct.unpack_from(chunk, sounding_offset + i * sounding_size)
		i+=1
			
		# THIS IS IT.  This is where we output xyz-points
		# Depths are referred to the reference point.  To get it to the waterline,
		# SUBSTRACT the distance from
		# Error estimates are also available: detectionUncertaintyVer_m and
		# detectionUncertaintyHor_m
		waterlevel = z_reRefPoint_m - z_waterLevelReRefPoint_m
		plat = latitude_deg + deltaLatitude_deg
		plon = longitude_deg + deltaLongitude_deg
		outputstr = f" %.{XY_DECIMALS}f %.{XY_DECIMALS}f %.2f %.2f" % (deltaLatitude_deg, deltaLongitude_deg, 
			z_reRefPoint_m, reflectivity1_dB)
		outputIO.write(outputstr)
		n = float(latitude_deg)
		e = float(longitude_deg)
		t = int(millisec)
		if (start_t < 0 or t < start_t):
			start_t = t
		if (t > stop_t):
			stop_t = t
		if (min_e > e):
			min_e = e
		if (min_n > n):
			min_n = n
		if (e > max_e):
			max_e = e
		if (n > max_n):
			max_n = n	

# Datagram processing follows the official structure given by Kongsberg
def processDatagram(lengtha, chunk, outputIO):
	header_without_length = struct.Struct('ccccBBHII')
	dgm_type0, dgm_type1, dgm_type2, dgm_type3, dgm_version, sysid, emid, sec, nsec = header_without_length.unpack_from(chunk, 0)
	dgm_type = dgm_type0 + dgm_type1 + dgm_type2 + dgm_type3
		
	# Decode time
	nanosec = sec
	nanosec *= 1E9
	nanosec += nsec
	millisec = nanosec
	millisec /= 1E6	
	
	# Decode datagram type/version
	strk = dgm_type.decode()
	if (strk == '#MRZ'):
		assert dgm_version == 3, "Wrong version of datagram, see Kongsberg library for KMALL files for original implementation"
		if (dgm_version == 3):
			process_MRZ_data(millisec, lengtha, chunk, outputIO)