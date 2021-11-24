; ------------------------------------------
; Script     : Traking _Softwere.ncl
; Updated on : 12 Sep, 2020 8:47:49 PM 
; Author     : Pradeep and Chetan
; ------------------------------------------
; Description:
;   Purpose of the script follows.
; ------------------------------------------
; This is the programe is the implimentation of cyclone tracking scheme from following study.
 
;1) Surface cyclones in the ERA40 data set (1958–2001). Part 1: Novel identification method and global climatology.J. from  for tracking of cyclones from spatical maps of the data
;    Wernli H, Schwierz C. (2006)
;2) Objective identification and tracking of multicentre cyclones in the ERA-Interim reanalysis dataset (Hanley, John Caballero, Rodrigo, 2012).
;
;3)use in ""Objective climatology of cyclones in the
;Mediterranean region: a consensus view among
;methods with different system identification and
;tracking criteria"
;
; This is writtern by Pradeep Kushwaha From Indian Institute of Science under his PHD
; Hugly benifited by efficint looping method impliment by Cheten kumar

begin 
DataPath         = "/home/pradeep/Desktop/MTC_PAPER2/AUTOMATIC_TRACKING_DATA/" 
;========================================================================
;                  call for the detection data from detection method
;========================================================================

CENTERS_DATA      =  addfile(DataPath+"CYCLONE_CENTERS_ALL_1998_2019.nc","r")

VOM_MD            =  CENTERS_DATA->VOR_RECORD(1,:)
VOL_MD            =  CENTERS_DATA->VOR_RECORD(0,:)
VOU_MD            =  CENTERS_DATA->VOR_RECORD(2,:)
QM_MD             =  CENTERS_DATA->Q_RECORD     
lon_md            =  CENTERS_DATA->lon
lat_md            =  CENTERS_DATA->lat
lev_md            =  CENTERS_DATA->lev
time_MD           =  CENTERS_DATA->time
LAT_RECORD        =  CENTERS_DATA->LAT_RECORD 
LON_RECORD        =  CENTERS_DATA->LON_RECORD
minima_MD         =  CENTERS_DATA->NUM_MIN
TIME_RECORD       =  CENTERS_DATA->TIME_RECORD
NUMBER_MIN        =  doubletoint( CENTERS_DATA->NUM_MIN  )
LEV_VMAX_RECO     =  CENTERS_DATA->LEV_VMAX_RECO
DIFF_MD           = VOM_MD-VOL_MD
print(cd_calendar(time_MD,3))
;===========================loop for saperate years====================
dates1 = (/1998060100,1999060100,2000060100,2001060100,2002060100,2003060100,2004060100,2005060100,2006060100,2007060100,2008060100,2009060100,2010060100,2011060100,2012060100,2013060100,2014060100,2015060100,2016060100,2017060100,2018060100,2019060100/)

dates2 = (/1998061018,1999093018,2000093018,2001093018,2002093018,2003093018,2004093018,2005093018,2006093018,2007093018,2008093018,2009093018,2010093018,2011093018,2012093018,2013093018,2014093018,2015093018,2016093018,2017093018,2018093018,2019093018/)
;======================================================================

;===========================loop for saperate years====================
;dates1 = (/1998060100,1998060400,1998060412/)
;dates2 = (/1998060318,1998060406,1998060418/)
;======================================================================
do dd = 0,0;dimsizes(dates1)-1

;=================loop for every year===================================
TIME_yyyymmddhh    = cd_calendar(time_MD,3)
LOOP_END_INDEX1    = ind(TIME_yyyymmddhh.eq.dates1(dd))
LOOP_END_INDEX2    = ind(TIME_yyyymmddhh.eq.dates2(dd))
;=======================================================================


;***************************************************
;           Temp stores trajectories
;           Added by Chetankumar Jalihal 
;           Upgraded by Pradeep Kushwaha 
;***************************************************

Temp           = new( (/dimsizes(LAT_RECORD), 9/), double)

TREJ           = new( (/dimsizes(LAT_RECORD), 9, 300/), double)

cy_index       = 0      ;Trejectory index
counter        = 0
Day_counter    = 0
Damping        = 1
npts           = 2
dist_threshold = 500   ; distance threshold
vor_threshold  = 5   ; Only defined, but not used

;study_period   = LOOP_END_INDEX;dimsizes( time_MD )       ; total number of days in the study
;print(study_period)
;return
do  i = LOOP_END_INDEX1,  LOOP_END_INDEX2                ; stop 11 days before the end of study period
;print(i)
    no_current_cy =  NUMBER_MIN(i)        ; No. of minima in the current day

    do j = 0, no_current_cy-1             ; for each minima do the following

            if (i.eq.LOOP_END_INDEX1)
                no_previous_cy := 0
            else
                no_previous_cy := sum( NUMBER_MIN(LOOP_END_INDEX1:i-1) )
            end if

           if VOM_MD(no_current_cy+j).gt.1.0 then ; consider a minima only if its vorticity is larger then 1.5
            ; Consider a minima
            Temp_time        = cd_calendar(TIME_RECORD( no_previous_cy + j), 3 )
            Temp_date        = TIME_RECORD( no_previous_cy + j)
            Temp_lat         = LAT_RECORD( no_previous_cy + j )
            Temp_lon         = LON_RECORD( no_previous_cy + j )
            Temp_diff        = DIFF_MD(no_previous_cy + j)
            Temp_humidity    = QM_MD(no_previous_cy + j) 
            Temp_vom         = VOM_MD(no_previous_cy+j)    
            Temp_si_lev      = LEV_VMAX_RECO(no_previous_cy+j)       

         print_clock(" WORKING ON "+Temp_time+" "+LOOP_END_INDEX1+" "+LOOP_END_INDEX2+" ")
            else
              continue
            end if   
            
            ; if the trajectory was identified previously (associated with another trajectory) then go to next cyclone
           
            if any(  (Temp_time.eq.Temp(:,1)) .and. (Temp_lat.eq.Temp(:,2)) .and. (Temp_lon.eq.Temp(:,3)) )
              
                continue
              
            end if
            
              ;---------------------------------------------------------
              ; if any trejectory start at 30 sep stop it 
                   TIMEI     = cd_calendar(Temp_date, 0)         
                   yearI    = toint( TIMEI(:,0) )
                   monthI   = toint( TIMEI(:,1) )
                   dayI     = toint( TIMEI(:,2) ) 
                       
                   if (monthI.eq.9).and.(dayI.eq.30) then
                      continue   
                   end if   
               ;----------------------------------------------------------
                           
               ; New trajectory
               
                cy_index     = cy_index + 1

                Temp(counter, 0) = cy_index
                Temp(counter, 1) = Temp_time
                Temp(counter, 2) = Temp_lat
                Temp(counter, 3) = Temp_lon
                Temp(counter, 4) = Temp_diff
                Temp(counter, 5) = Temp_humidity
                Temp(counter, 6) = Temp_vom
                Temp(counter, 7) = Temp_si_lev
                
               ;----------------------------------------------- 
               TREJ(cy_index-1, 0, 0)  = (/  cy_index       /) ; index of the track or track number
               TREJ(cy_index-1, 1, 0)  = (/  cd_calendar(Temp_date,3)      /) ; time string of the track point
               TREJ(cy_index-1, 2, 0)  = (/  Temp_lat       /) ; latitude of the center
               TREJ(cy_index-1, 3, 0)  = (/  Temp_lon       /) ; longitude of center
               TREJ(cy_index-1, 4, 0)  = (/  Temp_diff      /) ; middle - lower vorticity
               TREJ(cy_index-1, 5, 0)  = (/  Temp_humidity  /) ; mean middle troposphere humidity
               TREJ(cy_index-1, 6, 0)  = (/  Temp_vom       /) ; mean middle level vorticity
               TREJ(cy_index-1, 7, 0)  = (/  Temp_si_lev    /) ; level of vorticity maximum 
               TREJ(cy_index-1, 8, 0)  = (/    1            /) ; track length
               ;------------------------------------------------
                ; Track this minima for the NEXT 15 days
                do k = 0, 60  ; Randomly chosen 15 days, because we know cyclones won't last so long


                    ; identify location of minima for all cyclones
                    LAT_STEP2   := LAT_RECORD( sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1 )
                    LON_STEP2   := LON_RECORD( sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1 )
                    time_step2  := TIME_RECORD( sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1 )
                     vor_step2  := VOM_MD( sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1 )
                humidity_step2  := QM_MD( sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1)
                diff_md_step2   := DIFF_MD( sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1)
                si_p_max_step2  := LEV_VMAX_RECO(sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k))): sum(NUMBER_MIN(LOOP_END_INDEX1:(i+k+1)))-1)
                
                
                ; first gauss if initial point (k=0);
                ;-------------------------------------------------------------------------------------------------
                if k.eq.0 then
                  Temp_lat_guess := Temp_lat 
                  Temp_lon_guess := Temp_lon
                 else  ; linear interpolation of first gauss from second point onwards:
                  Temp_lat_guess := Temp_lat + Damping*(Temp(counter,2)-Temp(counter-1,2))
                  Temp_lon_guess := Temp_lon + Damping*(Temp(counter,3)-Temp(counter-1,3))
                end if 
                 ;--------------------------------------------------------------------------------------------------
                  if k.eq.0 then ; initial pridicted vorticity at first point
                  
                  Temp_vom_guess := Temp_vom ; 

                 else            ; pridicted vorticity for track older the 24 hours  
                 
                  Temp_vom_guess := Temp_vom + 0.75*(Temp(counter,6)-Temp(counter-1,6)) ; for tracks older then one day take linear projection of vorticity

                 end if 
                 
                  ;-------------------------------------------------------------------------------------------------- 
                 gcdist         := gc_latlon(Temp_lat_guess, Temp_lon_guess, LAT_STEP2, LON_STEP2, npts,4)
                 min_index      := minind( gcdist ) 
                 
                  ; print(dimsizes(min_index))
                  ;--------------------------------------------------------------------------------------------------
                  ;  THIS IS A SUBROUTINE TO TO SELECT THE NEXT CYCLONE WHICH ARE CLOSE TO VORTICITY OF CURRENT CYCLONE
                  ;---------------------------------------------------------------------------------------------------
                 if dimsizes(min_index).gt.1 then
                   
                   min_index     :=  minind(abs(Temp_vom-vor_step2(min_index) ) )
                   
                 else 
                   min_index     := min_index

                 end if 
                  ;---------------------------------------------------------------------------------------------------
                 
                 VOR_DT          := Temp_vom_guess-vor_step2(min_index) ; this is the minization of vorticity
                 
                  ;---------------------------------------------------------------------------------------------------  
                  ; Minimum distance and vorticity intensity criteria
                  ; only update this poin in track if it minimizes the coast function otherwise terminate it 
                   VORTICITY_THRESHOLD = (abs(VOR_DT).lt.vor_threshold)
                    if ((VORTICITY_THRESHOLD).and.(min(gcdist).le.dist_threshold).and.(vor_step2(min_index).gt.1.0) )

                    ; MINIMA MUST OF AT MINIMUM DISTENCE
                    ; MINIMA MUST OF OF SIMILAR INTENSITY (VOR*-VOR2<SI0)
                    ; AND MINIMA MUST BE ATLEST GREATER THEN 1 S-1     


                           
                     ;------------------------------------------------------------------------------------------------
                     ;                        Adding a small number for interpolation if cyclone is stationary
                     ;------------------------------------------------------------------------------------------------
                 
                         if (LAT_STEP2(min_index).eq.Temp(counter, 2)) then 
                         
                           LAT_STEP2(min_index) =  (LAT_STEP2(min_index))  ; spline works for different ponts
                           
                         end if 
                 
                 
                         if (LON_STEP2(min_index).eq.Temp(counter, 3)) then
                          
                           LON_STEP2(min_index) = (LON_STEP2(min_index))
                           
                         end if 
                     ;---------------------------------end small number addition-----------------------------------------
                           
                            Temp_lat            = (/LAT_STEP2(min_index)/)
                            Temp_lon            = (/LON_STEP2(min_index)/)
                            Temp_time           = (/cd_calendar( time_step2(min_index), 3)/)
                            Temp_date           = (/time_step2(min_index)/)
                            Temp_vom            = (/vor_step2(min_index)/)
                            Temp_vordiff        = (/diff_md_step2(min_index)/)
                            humidity_step2      = (/humidity_step2(min_index)/)
                            si_p_max_step2      = (/si_p_max_step2(min_index)/)
                      ; If it was previously tracked (associated with another minima) then move to next day
                      ;---------------------------------------------------------------------------------------------------   
                          if any(  (Temp_time.eq.Temp(:,1)) .and. (Temp_lat.eq.Temp(:,2)) .and. (Temp_lon.eq.Temp(:,3)) )
                                    break
                          end if
                       ;--------------------------------------------------------------------------------------------------
                            counter   = counter + 1 ; if minima minizes the coast function then update the trejectory

               TREJ(cy_index-1, 0, k+1)  = (/  cy_index                 /) ; index of the track or track number
               TREJ(cy_index-1, 1, k+1)  = (/  cd_calendar(time_step2(min_index), 3)   /) ; time string of the track point
               TREJ(cy_index-1, 2, k+1)  = (/ LAT_STEP2(min_index)      /) ; latitude of the center
               TREJ(cy_index-1, 3, k+1)  = (/ LON_STEP2(min_index)      /) ; longitude of center
               TREJ(cy_index-1, 4, k+1)  = (/ diff_md_step2(min_index)  /); middle - lower vorticity
               TREJ(cy_index-1, 5, k+1)  = (/ vor_step2(min_index)      /); mean middle troposphere humidity
               TREJ(cy_index-1, 6, k+1)  = (/ humidity_step2(min_index) /) ; mean middle level vorticity
               TREJ(cy_index-1, 7, k+1)  = (/ si_p_max_step2(min_index) /) ; level of vorticity maximum 
               TREJ(cy_index-1, 8, 0)    = (/    k+2                    /) ; track length
;---------------------------------------------------------------------------------------------------
                            Temp(counter, 0) = cy_index
                            Temp(counter, 1) = (/cd_calendar( time_step2(min_index), 3)/)
                            Temp(counter, 2) = (/LAT_STEP2(min_index)      /)
                            Temp(counter, 3) = (/LON_STEP2(min_index)      /)
                            Temp(counter, 4) = (/diff_md_step2(min_index)  /)
                            Temp(counter, 5) = (/vor_step2(min_index)      /)
                            Temp(counter, 6) = (/humidity_step2(min_index) /)
                            Temp(counter, 7) = (/si_p_max_step2(min_index) /)
                            Temp(counter, 8) = (/    k+2                   /)
                    else
                         counter = counter + 1  ; update new track point position if new trejectory 

                         break                  ; if coast function not minimum then stop the track 
                         
                    end if                      ; end of coast function loop 
                    
                    
           ;break if this is the last day of september
           
                   TIME    = cd_calendar(Temp_date, 0)         
                   year    = toint( TIME(:,0) )
                   month   = toint( TIME(:,1) )
                   day     = toint( TIME(:,2) ) 
                       
                   if (month.eq.10).and.(day.eq.30) then
                     break 
                   end if 
            
                 end do                         ; end of perticulat track update  


    
      
    end do                                      ; end of each minimum track initlization if not included in other trejectories
    
  
end do                                        ; end of days loop 


 basePath              =  "/home/pradeep/Desktop/MTC_PAPER2/AUTOMATIC_TRACKING_DATA/"
  pathOut              =  basePath + "AUTOMATIC_TRACKING_TRACKS_TEST_"+dates1(dd)+"_"+dates2(dd)+".nc"
  makeNewFile          =  True 
   setfileoption("nc", "Format",  "NetCDF4Classic")

 ;   setfileoption( "nc", "Format", "LargeFile" )
    if( makeNewFile ) then
    system( "rm " + pathOut )
    fout  = addfile( pathOut, "c" )

    ;===================================================================
    ; create global attributes of the file (optional)
    ;===================================================================
       fAtt               = True            ; assign file attributes
       fAtt@title         = "NCL Simple Approach to netCDF Creation 0=INDEX;1"
       fAtt@source_file   =  "original-file.nc"
       fAtt@Conventions   = "None"
       fAtt@creation_date = systemfunc ("date")
       fileattdef(fout , fAtt )            ; copy file attributes

    ;===================================================================
    ; make time an UNLIMITED dimension; recommended  for most applications
    ;===================================================================
     
       filedimdef(fout,"LAT",-1,True)
  
;--------------------------------------------------------------------------
   INDEX_ALL               =  TREJ(:,1,0)
   INDEX_ALL@_FillValue    =  -999
   delete( INDEX_ALL@_FillValue) ; this part is only to extract real values not missing
   INDEX_FINAL_MISSING     = ind(INDEX_ALL.eq.-999) 
   INDEX_FINAL             = INDEX_FINAL_MISSING(0)
  TRACK_LENGTH0            = TREJ(0:INDEX_FINAL,8,0)
  TRACK_IND24              = ind(TRACK_LENGTH0.ge.4) 
   TRAJ_24H                = TREJ(TRACK_IND24,:,:)
;-------------------------------------------------------------------------
   fout->TREJ                     = TRAJ_24H(:,:,:)
   fout->INDX_RECORD              = TRAJ_24H(:,0,:)
   fout->TIME_RECORD              = TRAJ_24H(:,1,:)
   fout->LAT_RECORD               = TRAJ_24H(:,2,:)
   fout->LON_RECORD               = TRAJ_24H(:,3,:)
   fout->DIFF_RECORD              = TRAJ_24H(:,4,:)
   fout->VOM_RECORD               = TRAJ_24H(:,5,:)
   fout->Q_RECORD                 = TRAJ_24H(:,6,:)
   fout->PMAX_RECORD              = TRAJ_24H(:,7,:)
   fout->TRACK_LENGTH             = TRAJ_24H(:,8,:)
  else
    fout  = addfile( pathOut, "w" )
  end if
;--------------------------------------------------------------------------
print_clock("WRIGHTING OVER FOR ALL MINIMA" )
;==========================================================================
alist = [/Temp(:,0), Temp(:,1), Temp(:,2), Temp(:,3), Temp(:,4), Temp(:,5), Temp(:,6), Temp(:,7)/]
write_table("Temp_file_2000_2004.txt", "w", alist, "%f%f%f%f%f%f%f%f%f")
;print(TREJ(:,8,0))
delete(TRAJ_24H)
delete(TREJ)
delete(Temp)
delete(INDEX_FINAL_MISSING)
delete(INDEX_FINAL)
delete(TRACK_LENGTH0)
delete(TRACK_IND24)


print("JOB SUCCESS")


end do 

end
;==============================================================================
