; This is the programe for tracking of cyclones from spatical maps of the data 
; This is writtern by Pradeep Kushwaha From Indian Institute of Science under his PHD


begin 
DataPath         = "/home/pradeep/Desktop/MTC_PAPER1/DATA_DERIVED/" 
;========================================================================
;                  call for the detection data from detection method
;========================================================================

CENTERS_DATA     =  addfile(DataPath+"MD_TC_NEW_COMPOSITS_QTWHUV.nc","r")
VOR_MD           =  CENTERS_DATA->VOR_COM
H_MD             =  CENTERS_DATA->H_COM
Q_MD             =  CENTERS_DATA->Q_COM
U_MD             =  CENTERS_DATA->U_COM
V_MD             =  CENTERS_DATA->V_COM
VOM_MD           =  CENTERS_DATA->VOR_RECORD(1,:)
VOL_MD           =  CENTERS_DATA->VOR_RECORD(0,:)
VOU_MD           =  CENTERS_DATA->VOR_RECORD(2,:)
QM_MD            =  CENTERS_DATA->Q_RECORD     
lon_md           =  CENTERS_DATA->lon
lat_md           =  CENTERS_DATA->lat
lev_md           =  CENTERS_DATA->lev
time_MD          =  CENTERS_DATA->time
LAT_RECORD       =  CENTERS_DATA->LAT_RECORD 
LON_RECORD       =  CENTERS_DATA->LON_RECORD
minima_MD        =  CENTERS_DATA->minima
TIME_RECORD      =  CENTERS_DATA->TIME_RECORD
NUMBER_MIN       =  CENTERS_DATA->NUM_MIN 


INDEX_FILTER     = ind(VOM_MD.gt.-100)

LAT_RECORD_MD    = LAT_RECORD(INDEX_FILTER)
LON_RECORD_MD    = LON_RECORD(INDEX_FILTER)
TIME_RECORD_MD   = TIME_RECORD(INDEX_FILTER)
;============================================================================
;                   DEFINE VARIABLES TO STORE THE TREJECTORIES 
;============================================================================

MAX_NUM_TREJ    = 500

MAX_LENGTH_TEJ  = 100

LAT_TRJ         = new((/MAX_NUM_TREJ,MAX_LENGTH_TEJ/),"float")

LON_TRJ         = new((/MAX_NUM_TREJ,MAX_LENGTH_TEJ/),"float")

VORM_TRJ        = new((/MAX_NUM_TREJ,MAX_LENGTH_TEJ/),"float")

VORL_TRJ        = new((/MAX_NUM_TREJ,MAX_LENGTH_TEJ/),"float")

VOR_DIFF_TRJ    = new((/MAX_NUM_TREJ,MAX_LENGTH_TEJ/),"float")

 IND_TRJ        = new(MAX_NUM_TREJ,"integer")

 IND_TRJ0       = new(1,float)
;=============================================================================
;                   EXTRACT PER DAY CYCLONIC CENTERS
;=============================================================================
TIMEIN   =  30
NMIN     = toint(NUMBER_MIN)
NCSUM    = cumsum(NMIN,2)  
;----------------------------------------------------------------------------
;NMIN is the variable which stores the number of detected centers per step
; 1 june 2014 we have 13 min and so on..
;---------------------------------------------------------------------------
do T = 0,500; dimsizes(NMIN)-2

;----------------------------------------------------------------------------
; In this step we are extracting sets of minimum during two analysis time steps
; cumulaive sum provides the indexs for next dates in the lat lon indexes.
;----------------------------------------------------------------------------
;if T.EQ.0 then
;----------------------------------------------------------------------------
;                      Initliz the trejectories------------------------------
;LAT_STEP1  = LAT_RECORD_MD(0:NMIN(0))  
;LON_STEP1  = LON_RECORD_MD(0:NMIN(0))
;---------------------------------------------------------------------------
;print(NMIN(0))
;**************************************************************************
  if T.eq.0 then 
      IND_ST_STEP1    = 0
      IND_EN_STEP1    = NMIN(0)-1

      IND_ST_STEP2     = cumsum(NCSUM(T),2)
      IND_EN_STEP2     = cumsum(NCSUM(T+1),2)-1
;;--------------------------------------------------------------------
   LAT_STEP1          = LAT_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1)  
   LON_STEP1          = LON_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1)  
;----------------------------------------------------------------------
   LAT_STEP2          = LAT_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2)  
   LON_STEP2          = LON_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2)


   TIME_STEP1         = TIME_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1)
   TIME_STEP2         = TIME_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2)

   VOR_STEP1          =  VOM_MD(IND_ST_STEP1:IND_EN_STEP1)  
   VOR_STEP2          =  VOM_MD(IND_ST_STEP2:IND_EN_STEP2)  

;print(""+LAT_STEP1+"  " +   "  "+LON_STEP1+"    "+cd_calendar(TIME_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1),2)+"")

;print(""+LAT_STEP2+"  " +   "  "+LON_STEP2+"    "+cd_calendar(TIME_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2),2)+"")
;print("end of T=0")
;----------------------------------------------------------------------
   else
;--------------------------------------------------------------------
     IND_ST_STEP1     = cumsum(NCSUM(T-1),2)
     IND_EN_STEP1     = cumsum(NCSUM(T),2)-1
 
     IND_ST_STEP2     = cumsum(NCSUM(T),2)
     IND_EN_STEP2     = cumsum(NCSUM(T+1),2)-1
;;------------------------------------------------------------------
     LAT_STEP1        = LAT_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1)  
     LON_STEP1        = LON_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1)  

     LAT_STEP2        = LAT_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2)  
     LON_STEP2        = LON_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2)

     TIME_STEP1       = TIME_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1)
     TIME_STEP2       = TIME_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2)
;print(""+LAT_STEP1+"  " +   "  "+LON_STEP1+"    "+cd_calendar(TIME_RECORD_MD(IND_ST_STEP1:IND_EN_STEP1),2)+"")

;print(""+LAT_STEP2+"  " +   "  "+LON_STEP2+"    "+cd_calendar(TIME_RECORD_MD(IND_ST_STEP2:IND_EN_STEP2),2)+"")

  ; VOR_STEP1          =  VOM_MD(IND_ST_STEP1:IND_EN_STEP1)  
  ; VOR_STEP2          =  VOM_MD(IND_ST_STEP2:IND_EN_STEP2)  
;----------------------------------------------------------------------
   end if 


;========================================================================================

;                                      DISTENCE CALCULATION
;-----------------------------------------------------------------------------------------
;                           -DEFINE FEW CONSTENT FOR DISTENCE CALCULATION

DIST0   = 1000 
npts    = 2
;-----------------------------------------------------------------------------------------
TF      = 0
TNF     = 0
track   = 0
;-----------------------------------------------------------------------------------------
;printVarSummary(TIME_RECORD_MD)
;===============================INITILIZATION OF TRACKS====================================

;----------------------------------NUM  MIN IN STEP 1-------------------------------------
    TSTEP1_NMIN           = dimsizes(LAT_STEP1)
;----------------------------------NUM  MIN IN STEP 2-------------------------------------
    TSTEP2_NMIN           = dimsizes(LAT_STEP2)

    TRACK_LAT_STORE1      = new(dimsizes(LAT_STEP1),float)
    TRACK_LON_STORE1      = new(dimsizes(LAT_STEP1),float)

    TRACK_LAT_STORE2      = new(dimsizes(LAT_STEP1),float)
    TRACK_LON_STORE2      = new(dimsizes(LAT_STEP1),float)


    TRACK_INDEX_STORE1    = new(dimsizes(LAT_STEP1),float)
    TRACK_INDEX_STORE2    = new(dimsizes(LAT_STEP1),float)
  

    TRACK_TIME_STORE1     = new(dimsizes(LAT_STEP1),double)
    TRACK_TIME_STORE2     = new(dimsizes(LAT_STEP1),double)

do NN = 0, TSTEP1_NMIN-1

  if T .eq.0 then

   
;==================================================================================================
 

        gcdist              = gc_latlon(LAT_STEP1(NN), LON_STEP1(NN), LAT_STEP2, LON_STEP2, npts,4)
;-------------------------------------------------------------------------------------------------
;           calculate minimum distence candidate-for each initial point
;--------------------------------------------------------------------------------------------------
        DISTMIN             = min(gcdist)
        IND_MIN             = ind(gcdist.eq.DISTMIN)

        LAT_NEX             = LAT_STEP2(IND_MIN)
        LON_NEX             = LON_STEP2(IND_MIN)
 

        LAT_CUR             = LAT_STEP1(NN)
        LON_CUR             = LON_STEP1(NN) 

        TIME_CUR            = TIME_STEP1(NN)
        TIME_NEX            = TIME_STEP2(IND_MIN(0))  ; (********************************NEED TO DEAL WITH IT*********************************) 
;--------------------                                 ---filtering out systems with minimum distence critaria----------------------------------
       if DISTMIN.lt.DIST0 then

        TF     = TF+1

        TNF    = TNF+1

        TRACK_LAT_STORE1(track)   = tofloat(LAT_CUR)
        TRACK_LON_STORE1(track)   = tofloat(LON_CUR)

        TRACK_LAT_STORE2(track)   = tofloat(LAT_NEX)
        TRACK_LON_STORE2(track)   = tofloat(LON_NEX)


        TRACK_INDEX_STORE1(track) = toint(NN)
        TRACK_INDEX_STORE2(track) = toint(IND_MIN(0))


        TRACK_TIME_STORE1(track) = TIME_CUR
        TRACK_TIME_STORE2(track) = TIME_NEX





;print(""+NN+"  "  +  "  "+IND_MIN+"")

       ;  print(DISTMIN.lt.DIST0)
      else
    
        TRACK_LAT_STORE1(track)   = 1000 ; TRACK_LAT_STORE1@_FillValue
        TRACK_LON_STORE1(track)   = 1000 ; TRACK_LAT_STORE1@_FillValue


        TRACK_LAT_STORE2(track)   = 1000;  TRACK_LAT_STORE1@_FillValue ; CUR : CURRENT
        TRACK_LON_STORE2(track)   = 1000; TRACK_LAT_STORE1@_FillValue


        TRACK_INDEX_STORE1(track) = 1000; TRACK_LAT_STORE1@_FillValue
        TRACK_INDEX_STORE2(track) = 1000; TRACK_LAT_STORE1@_FillValue


        TRACK_TIME_STORE1(track) =  TIME_CUR
        TRACK_TIME_STORE2(track) =  TIME_NEX



        NGEN_POINTS   =  TSTEP2_NMIN-TF             ; NGEN ARE THE NEW GENSIS POINTS AT NEXT TIME STEP WHICH IS TOTAL =  (POINTS AT STEP2 - POINTS WHO GOT TRACK)

      end if     

        TRACK_FOUND   = TF
        TRACK_NFOUND  = TNF 
        TRACK_NLAT    = LAT_NEX 
        TRACK_NLON    = LON_NEX
 
    

  track = track + 1                               ; increament of t 0 tracks

;===========================================================================================================================================
;                                                   strat for next time step grater the 0
;===========================================================================================================================================
 else                                             ; start of T>0 condition  and  the end of T= 0 condition 

        gcdist              = gc_latlon(LAT_STEP1(NN), LON_STEP1(NN), LAT_STEP2, LON_STEP2, npts,4)
;-------------------------------------------------------------------------------------------------------------------------------------------
;                                                   calculate minimum distence candidate-for each initial point
;--------------------------------------------------------------------------------------------------------------------------------------------
        DISTMIN             = min(gcdist)
;--------------------------------------------------------------------------------------------------------------------------------------------


        IND_MIN             = ind(gcdist.eq.DISTMIN)

        LAT_NEX             = LAT_STEP2(IND_MIN(0))
        LON_NEX             = LON_STEP2(IND_MIN(0))

  
        LAT_CUR             = LAT_STEP1(NN)
        LON_CUR             = LON_STEP1(NN) 

        TIME_CUR            = TIME_STEP1(NN)
        TIME_NEX            = TIME_STEP2(IND_MIN(0))
;----------------------                          -filtering out systems with minimum distence critaria----------------------------------------
       if DISTMIN.lt.DIST0 then

        TF     = TF+1

        TNF    = TNF+1

        TRACK_LAT_STORE1(track)   = tofloat(LAT_CUR)
        TRACK_LON_STORE1(track)   = tofloat(LON_CUR)


        TRACK_LAT_STORE2(track)   = tofloat(LAT_NEX)    ; CUR : CURRENT
        TRACK_LON_STORE2(track)   = tofloat(LON_NEX)


        TRACK_INDEX_STORE1(track) = toint(NN)
        TRACK_INDEX_STORE2(track) = toint(IND_MIN(0))

  
        TRACK_TIME_STORE1(track) = TIME_CUR
        TRACK_TIME_STORE2(track) = TIME_NEX


      else
 
        NGEN_POINTS   =  TSTEP2_NMIN-TF            ; NGEN ARE THE NEW GENSIS POINTS AT NEXT TIME STEP WHICH IS TOTAL =  (POINTS AT STEP2 - POINTS WHO GOT TRACK)


        TRACK_LAT_STORE1(track)   =  TRACK_LAT_STORE1@_FillValue
        TRACK_LON_STORE1(track)   =  TRACK_LAT_STORE1@_FillValue


        TRACK_LAT_STORE2(track)   =  TRACK_LAT_STORE1@_FillValue ; CUR : CURRENT
        TRACK_LON_STORE2(track)   =  TRACK_LAT_STORE1@_FillValue


        TRACK_INDEX_STORE1(track) =  TRACK_LAT_STORE1@_FillValue
        TRACK_INDEX_STORE2(track) =  TRACK_LAT_STORE1@_FillValue


        TRACK_TIME_STORE1(track) =  TIME_CUR
        TRACK_TIME_STORE2(track) =  TIME_NEX



      end if     

        TRACK_FOUND   = TF
        TRACK_NFOUND  = TNF 
        TRACK_NLAT    = LAT_NEX
        TRACK_NLON    = LON_NEX


       track = track + 1                            ; increament of non T>0 tracks-----------------------------------------
end if                                               ; end of T condition as T=0  or T>0 ----------------------------------

delete(IND_MIN)

end do                                              ; end of number of assigining next candidate for maximim ar STEP1

;===================================no track condition need to deal========================================================



IND_TRACKS_STEP1          = ind(.not.ismissing(TRACK_INDEX_STORE1))  ;   INDEX CORROSPONDS TO PREVIOUS POINT 
IND_TRACKS_STEP2          = ind(.not.ismissing(TRACK_INDEX_STORE2))  ;   INDEX CORROSPONGS TO CURRENT  POINT

;print(TRACK_LAT_STORE2)
;print(IND_TRACKS_STEP2)
;print(TRACK_LAT_STORE2(IND_TRACKS_STEP2))
;==================================STORING OF TREJECTORIES AFTER EACH TIME STEP=============================================
if T.EQ.0 then 


  INDEX_TREJ1               = toint(TRACK_INDEX_STORE1(IND_TRACKS_STEP1))
  INDEX_TREJ2               = toint(TRACK_INDEX_STORE2(IND_TRACKS_STEP2))

;============================================================================================================================
;            LOOP CORROSPONDS TO TREJECTORY AT PRESENT TIME STEP
;============================================================================================================================


   ;-------------------------------------------------------------------------------------------------------------------------
   ; LAT_TRJ(IND_TRJ(0:TRACK_FOUND),0)   = TRACK_LAT_STORE1(IND_TRACKS_STEP1) ;TRACK_LAT_STORE1(toint(INDEX_TREJ1(TREJ)))
  ;  LON_TRJ(IND_TRJ(0:TRACK_FOUND),0)   = TRACK_LON_STORE1(IND_TRACKS_STEP1) ;TRACK_LON_STORE1(toint(INDEX_TREJ1(TREJ)))


  ;  LAT_TRJ(IND_TRJ(0:TRACK_FOUND-1),1)   = TRACK_LAT_STORE2(IND_TRACKS_STEP2) ;TRACK_LAT_STORE2(toint(INDEX_TREJ2(TREJ)))
  ;  LON_TRJ(IND_TRJ(0:TRACK_FOUND-1),1)   = TRACK_LAT_STORE2(IND_TRACKS_STEP2) ;TRACK_LON_STORE2(toint(INDEX_TREJ2(TREJ)))

;----------------------------------------------------------------------------------------------------------------------------
;print(""+TRACK_LAT_STORE1(IND_TRACKS_STEP1)+"  " +  "  "+TRACK_LON_STORE1(IND_TRACKS_STEP1)+"  "  +  "  "+TRACK_LAT_STORE2(IND_TRACKS_STEP2)+"  "   +  "  "+TRACK_LON_STORE2(IND_TRACKS_STEP2)+"")
else
;================================need to define trejectory indexes===========================================================
INDEX_TREJ1               = toint(TRACK_INDEX_STORE1(IND_TRACKS_STEP1))
INDEX_TREJ2               = toint(TRACK_INDEX_STORE2(IND_TRACKS_STEP2))
end if 

print(""+TRACK_LON_STORE1(IND_TRACKS_STEP1)+"  " +  "  "+TRACK_LON_STORE2(IND_TRACKS_STEP2)+"  "  +  "  "+cd_calendar(TRACK_TIME_STORE1(IND_TRACKS_STEP1),2)+"  ")
; T+1 LAT1 IS MACHING WITH T LAT2 AND SO ON........

;=============================================================================================================================
delete(INDEX_TREJ1)
delete(INDEX_TREJ2)
delete(IND_TRACKS_STEP1)
delete(IND_TRACKS_STEP2)
delete(TRACK_LAT_STORE1)   ; delete the lat track array to define new track array for next time step
delete(TRACK_LON_STORE1)   ; delete the lat track array to define new track array for next time step
delete(TRACK_INDEX_STORE1)

delete(TRACK_LAT_STORE2)   ; delete the lat track array to define new track array for next time step
delete(TRACK_LON_STORE2)   ; delete the lat track array to define new track array for next time step
delete(TRACK_INDEX_STORE2)
delete(TRACK_TIME_STORE1)
delete(TRACK_TIME_STORE2)
;------------------------------------------------------------------------------------------------------------------
delete(IND_ST_STEP1)
delete(IND_EN_STEP1)
delete(IND_ST_STEP2)
delete(IND_EN_STEP2)

delete(LAT_STEP1)
delete(LON_STEP1)
delete(LAT_STEP2)
delete(LON_STEP2)
delete(TIME_STEP1)
delete(TIME_STEP2)
delete(gcdist)

end do



print("JOB SUCCESS")

end
;==============================================================================
