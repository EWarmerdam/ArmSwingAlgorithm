function [ arm_swing ] = arm_swing_algorithm( ang_vel, fs, TH_min_ampl, varargin )
%arm_swing_algorithm Calculates several arm swing parameters from IMU(s) on
%the wrist during walking
%
%   Definitions:
%   swing = either a backward or a forward swing
%   cycle = 2 swings (backward and forward)
%
% toolboxes required to run the algorithm: signal toolbox and statistics
% toolbox
%**************************************************************************
%   input:
%          ang_vel     = raw angular velocity (rad/s) from IMU (x forward (direction of thumb), y left and z
%                        vertical in n pose (Nx3 when using one IMU and Nx6 when using two IMUs
%                        (first three channels left, last three right arm))
%          fs          = sample frequency of the IMU
%          TH_min_ampl = Amplitude threshold. Every swing below this
%                        threshold will be discarded
%          varargin    = Optional input parameter for the number of swings
%                        that should be removed at the start and end of the walk when
%                        only analysing steady state walking ([start, end])
%   output:
%           amplitude  = amplitude per swing (range of motion) [deg]
%           pk_ang_vel = peak angular velocity per swing [deg/s]
%           start_idx  = sample number at which a swing starts
%           end_idx    = sample number at which a swing ends
%     regularity_angle = regularity of the angular signal (similarity of
%                        neighbouring swings; 1 = similar to neighbouring swings) (0-1)
%   regularity_ang_vel = regularity of the angular velocity signal
%       pk_vel_forward = average peak angular velocity of all the forward swings
%      pk_vel_backward = average peak angular velocity of all the backward swings
%      perc_time_swing = time during walking that there were swings detected [%]
%            frequency = frequency of the arm cycle [Hz]
%      perc_both_swing = percentage time during walking bout that there is
%                        arm swing detected in both arms [%]
%  amplitude_asymmetry = asymmetry of the amplitude between left and right swings (0% means no asymmetry) [%]
%peak_velocity_asymmetry= asymmetry of the peak angular velocity between left and right swings [%]
%     coordination_max = coordination of the timing between left and right swings (1 when the arms
%                        move exactly out of phase with each other) (0-1)
%
%**************************************************************************
% Written by Elke Warmerdam, Kiel University,
% e.warmerdam@neurologie.uni-kiel.de


%% preprocessing

%filter data
N_lp = 2;
[b_lp,a_lp] = butter(N_lp,3/(.5*fs),'low');
ang_vel_filt = filtfilt(b_lp, a_lp, ang_vel);

ang_vel_deg = ang_vel_filt*180/pi;

% check whether one arm or both arms were measured
if size(ang_vel,2)==3
    nr_imu = 1;
elseif size(ang_vel,2)==6
    nr_imu = 2;
else
    error('the angular velocity does not have the correct amount of channels')
end

for i_imu = 1:nr_imu
    if nr_imu == 2 && i_imu == 1
        ang_vel_imu = ang_vel_deg(:,1:3);
    elseif nr_imu == 2 && i_imu == 2
        ang_vel_imu = ang_vel_deg(:,4:6);
    else
        ang_vel_imu = ang_vel_deg;
    end
    
    %% extract angle and angular velocity
    % calculate principle component
    [coeff_av, ang_vel_pca] = pca(ang_vel_imu(:,1:2)-nanmean(ang_vel_imu(:,1:2)));
    if  coeff_av(2,1) <0
        ang_vel_pca(:,1) = ang_vel_pca(:,1)*-1;
    end
    
    % calculate angle
    angle = (cumtrapz(ang_vel_pca)/fs);
    
    % moving average filter to remove trend in data
    wts = [1/(2*fs);repmat(1/fs,fs-1,1);1/(2*fs)];
    mov_avg = nan(length(angle)-fs,2);
    for i_ch = 1:2
        mov_avg(:,i_ch) = conv(angle(:,i_ch),wts,'valid');
    end
    angle_pca = angle(round(.5*fs)+1:length(angle)-(round(.5*fs)),:)- mov_avg;
    ang_vel_pca = ang_vel_pca(round(.5*fs)+1:length(angle)-(round(.5*fs)),:);
    
    %% extract dominant frequency
    window = 3*fs;
    steps = .25*window; % 75% overlap
    
    swing_freq = nan(round((length(angle_pca) - window)/steps) +1,1);
    tot_power = nan(round((length(angle_pca) - window)/steps) +1,1);
    power_band = nan(round((length(angle_pca) - window)/steps) +1,1);
    perc_power = nan(round((length(angle_pca) - window)/steps) +1,1);
    
    if window*1.5 < length(angle_pca)
        nfft = pow2(nextpow2(length(angle_pca(:,1))));
        for i_w = 1:(steps):(length(angle_pca) - window)
            y = fft((angle_pca(i_w:i_w + window,1)-nanmean(angle_pca(i_w:i_w + window,1))),nfft);
            y = abs(y.^2); % raw power spectrum density
            y = y(1:1+nfft/2); % half-spectrum
            f_scale = (0:nfft/2)* fs/nfft; % frequency scale
            st_f = find(f_scale>.3, 1, 'first'); % frequency range for arm swing .3-3 Hz
            end_f = find(f_scale> 3, 1, 'first');
            
            % search dominant frequency in PSD
            [pks_height,pk_idx] = findpeaks(y(st_f:end_f)); % find maximum between .3Hz and 3 Hz
            pk_idx = pk_idx + st_f -1;
            [~, dom_pk_idx] = max(pks_height);
            
            if length(pk_idx)>= 1
                swing_freq(round(i_w/steps) +1,1) = f_scale(pk_idx(dom_pk_idx));
            else
                swing_freq(round(i_w/steps) +1,1) = nan(1);
            end
            
            % define how much power there is in the .3-3Hz domain
            tot_power(round(i_w/steps) +1,1) = trapz(y);
            power_band(round(i_w/steps) +1,1)  = trapz(y(st_f:end_f));
            perc_power(round(i_w/steps) +1,1) = (power_band(round(i_w/steps) +1,1) / tot_power(round(i_w/steps) +1,1))*100;
        end
        perc_power(end,1) = perc_power(end-1,1);
        
        swing_freq(length(swing_freq):length(swing_freq)+1) = repmat(swing_freq(length(swing_freq)-1),2,1);
        
        % When percentage power in .3-3Hz domain is below 90 no clear arm
        % swing present (amplitude and peak angular velocity not calculated
        % in that case)
        
        TH = 90;
        idx_remove = perc_power < TH;
        
        %% extract peaks during windows
        % calculate cycle time
        window_pk = 3*fs;
        steps_pk = .5*window_pk; % 50% overlap
        cycle_time_k = nan(round((length(angle_pca) - window_pk)/steps_pk) +1,1);
        for i_wp = 1:(steps_pk):(length(angle_pca) - window_pk)
            if i_wp ==1
                cycle_time = 1/nanmedian(swing_freq(1:2));
                if sum(idx_remove(1:2))>0
                    cycle_time = nan;
                end
            elseif i_wp == floor(length(angle_pca)/steps_pk)*steps_pk - window_pk +1 %last window
                cycle_time = 1/ nanmean(swing_freq((round(i_wp/steps_pk) +1) *2 -2:end));
                if sum(idx_remove((round(i_wp/steps_pk) +1) *2 -2:length(idx_remove)))>1
                    cycle_time = nan;
                end
            else
                cycle_time = 1/ nanmean(swing_freq((round(i_wp/steps_pk) +1) *2 -2:(round(i_wp/steps_pk) +1) *2 +1));
                if sum(idx_remove(((round(i_wp/steps_pk) +1) *2 -2:(round(i_wp/steps_pk) +1) *2 +1))) >1
                    cycle_time = nan;
                end
            end
            
            % extract peaks in angle signal
            if ~isnan(cycle_time)
                [~, locs_neg_angle] = findpeaks(-angle_pca(i_wp:i_wp + window_pk,1),...
                    'MinPeakDistance',0.6 * cycle_time *fs, 'MinPeakProminence', 2);
                locs_neg_angle = locs_neg_angle + i_wp -1;
                
                all_locs_neg_angle(round(i_wp/steps_pk) +1,1:length(locs_neg_angle)) = locs_neg_angle';
                
                [~, locs_pos_angle] = findpeaks(angle_pca(i_wp:i_wp + window_pk,1),...
                    'MinPeakDistance',0.6 * cycle_time *fs, 'MinPeakProminence', 2);
                locs_pos_angle = locs_pos_angle + i_wp -1;
                
                all_locs_pos_angle(round(i_wp/steps_pk) +1,1:length(locs_pos_angle)) = locs_pos_angle';
            end
            
            cycle_time_k( round(i_wp/steps_pk) +1,1)= cycle_time;
        end
        
        if ~all(isnan(cycle_time_k))
            % remove peaks that were detected twice due to overlapping windows
            idx_neg_pks(1:length(unique(all_locs_neg_angle)),1) = unique(all_locs_neg_angle);
            idx_neg_pks(idx_neg_pks==0) = [];
            idx_pos_pks(1:length(unique(all_locs_pos_angle)),1) = unique(all_locs_pos_angle);
            idx_pos_pks(idx_pos_pks==0) = [];
            
            % make sure there is always a positive peak after a negative peak and
            % vice versa, if multiple peaks in between, delete lowest peak
            i_pks = 1;
            if idx_pos_pks(1)> idx_neg_pks(1)
                while i_pks < length(idx_neg_pks)-1 && i_pks < length(idx_pos_pks)
                    if idx_neg_pks(i_pks+1) <idx_pos_pks(i_pks)
                        if angle_pca(idx_neg_pks(i_pks+1)) < angle_pca(idx_neg_pks(i_pks))
                            idx_neg_pks(i_pks)=[];
                        else
                            idx_neg_pks(i_pks+1)=[];
                        end
                        i_pks = i_pks -1;
                    end
                    if i_pks >= 1 && idx_neg_pks(i_pks) >idx_pos_pks(i_pks)
                        if angle_pca(idx_pos_pks(i_pks)) > angle_pca(idx_pos_pks(i_pks-1))
                            idx_pos_pks(i_pks-1)=[];
                        else
                            idx_pos_pks(i_pks)=[];
                        end
                        i_pks = i_pks -1;
                    end
                    i_pks = i_pks +1;
                end
            elseif idx_pos_pks(1)< idx_neg_pks(1)
                while i_pks < length(idx_pos_pks)-1 && i_pks < length(idx_neg_pks)
                    if idx_pos_pks(i_pks+1) <idx_neg_pks(i_pks)
                        if angle_pca(idx_pos_pks(i_pks+1)) > angle_pca(idx_pos_pks(i_pks))
                            idx_pos_pks(i_pks)=[];
                        else
                            idx_pos_pks(i_pks+1)=[];
                        end
                        i_pks = i_pks -1;
                    end
                    if i_pks > 1 && idx_pos_pks(i_pks) >idx_neg_pks(i_pks)
                        if angle_pca(idx_neg_pks(i_pks)) < angle_pca(idx_neg_pks(i_pks-1))
                            idx_neg_pks(i_pks-1)=[];
                        else
                            idx_neg_pks(i_pks)=[];
                        end
                        i_pks = i_pks -1;
                    end
                    i_pks = i_pks +1;
                end
            end
            idx_neg_pks = idx_neg_pks(1:i_pks,1);
            idx_pos_pks = idx_pos_pks(1:i_pks,1);
            
            idx_all_pks = [idx_neg_pks;idx_pos_pks];
            idx_all_pks = sort(idx_all_pks);
            height_pks = angle_pca(idx_all_pks);
            
            %% option to remove first and last few swings
            if nargin ==4
                rmv_first_last = cell2mat(varargin);
                if length(idx_all_pks)<= sum(rmv_first_last)
                    warning('Could not remove first and last few swings. Not enough swings present.');
                    idx_all_pks = [];
                    height_pks = [];
                else
                    idx_all_pks(1:rmv_first_last(1)) = [];
                    idx_all_pks(length(idx_all_pks) - rmv_first_last(2) +1:length(idx_all_pks)) = [];
                    
                    height_pks(1:rmv_first_last(1)) = [];
                    height_pks(length(height_pks) - rmv_first_last(2) +1:length(height_pks)) = [];
                end
            end
            
            if ~isempty(idx_all_pks) && length(idx_all_pks) >1
                %% remove outliers
                % delete values above 3 times 80th percentile of peaks
                TH_outlier = prctile(abs(height_pks),80) *3;
                rmv_outlier = abs(height_pks)>TH_outlier;
                
                % also remove if there is one swing in between two outliers
                idx_outlier = find(rmv_outlier == 0);
                for i_o = 2: length(idx_outlier)-1
                    if rmv_outlier(idx_outlier(i_o)-1) == 1 &&  rmv_outlier(idx_outlier(i_o)+1) == 1
                        rmv_outlier(idx_outlier(i_o)) = 1;
                    end
                end
                
                % delete if there is a large time gap between peaks (no consecutive cycles)
                for i_d = 1:length(idx_all_pks)-1
                    dlt(i_d,1) = (idx_all_pks(i_d+1) - idx_all_pks(i_d))> 2*nanmean(cycle_time_k)*fs;
                end
                
                %check peak detection
%                 tt = (1:length(angle_pca))/fs;
%                 figure; subplot(2,1,1);plot(tt,angle_pca(:,1)); hold on;
%                 plot(tt(idx_neg_pks),angle_pca(idx_neg_pks),'>');plot(tt(idx_pos_pks),angle_pca(idx_pos_pks),'<');
%                 plot(tt(idx_all_pks(dlt==0)),angle_pca(idx_all_pks(dlt==0)),'*');
%                 plot(tt(idx_all_pks(rmv_outlier ==1)), angle_pca(idx_all_pks(rmv_outlier ==1)),'x','linewidth',5)
%                 xlabel('time (s)'); ylabel('angle (deg)'); title('detected peaks for calculation of RoM and peak angular velocity');
%                 subplot(2,1,2); plot(tt,ang_vel_pca(:,1));
%                 hold on; title('angular velocity')
%                 plot(tt(idx_all_pks(dlt==0)),ang_vel_pca(idx_all_pks(dlt==0)),'*');
%                 xlabel('time (s)'); ylabel('angular velocity (deg/s)');
                
                %% calculate parameters
                
                % calculate peak angular velocity and amplitude
                ampl = nan(length(idx_all_pks)-1,1);
                pk_ang_vel = nan(length(idx_all_pks)-1,1);
                for i_v = 1: length(idx_all_pks)-1
                    if height_pks(i_v)>0 && height_pks(i_v+1)<0 || height_pks(i_v)<0 && height_pks(i_v+1)>0
                        ampl(i_v,1) = abs(height_pks(i_v)) + abs(height_pks(i_v+1));
                    else
                        if abs(height_pks(i_v)) < abs(height_pks(i_v+1))
                            ampl(i_v,1) = abs(height_pks(i_v+1) - height_pks(i_v));
                        else
                            ampl(i_v,1) = abs(height_pks(i_v) - height_pks(i_v+1));
                        end
                    end
                    
                    if ismember(idx_all_pks(i_v), idx_pos_pks)
                        pk_ang_vel(i_v,1) = abs(min(ang_vel_pca(idx_all_pks(i_v):idx_all_pks(i_v+1),1)));
                    else
                        pk_ang_vel(i_v,1) = abs(max(ang_vel_pca(idx_all_pks(i_v):idx_all_pks(i_v+1),1)));
                    end
                end
                
                pk_ang_vel(pk_ang_vel>500)= nan;
                
                start_idx(:,1) = idx_all_pks(1:end-1);
                end_idx(:,1) = idx_all_pks(2:end);
                
                % delete swings with outliers and non consecutive swings
                idx_rmv_outlier = find(rmv_outlier==1);
                idx_dlt = find(dlt==1);
                if ~isempty(idx_rmv_outlier) && ~isempty(idx_dlt)
                    rmv  = sort([(idx_rmv_outlier-1);idx_rmv_outlier; idx_dlt]);
                elseif ~isempty(idx_rmv_outlier) && length(idx_dlt)<1
                    rmv  = sort([(idx_rmv_outlier-1);idx_rmv_outlier]);
                elseif length(idx_rmv_outlier)<1 && ~isempty(idx_dlt)
                    rmv  =  idx_dlt;
                else
                    rmv = nan;
                end
                
                if rmv(end)> length(ampl)
                    rmv(end) = [];
                end
                
                if ~isnan(rmv)
                    rmv(rmv==0) = [];
                    ampl(rmv) = [];
                    pk_ang_vel(rmv) = [];
                    start_idx(rmv) = [];
                    end_idx(rmv) = [];
                end
                
                % delete all swings with an amplitude below a threshold
                pk_ang_vel(ampl<TH_min_ampl) = [];
                start_idx(ampl<TH_min_ampl) = [];
                end_idx(ampl<TH_min_ampl) = [];
                ampl(ampl<TH_min_ampl) = [];
                
                % delete swings with a peak ang vel below 10
                TH_min_pav = 10;
                start_idx(pk_ang_vel<TH_min_pav) = [];
                end_idx(pk_ang_vel<TH_min_pav) = [];
                ampl(pk_ang_vel<TH_min_pav) = [];
                pk_ang_vel(pk_ang_vel<TH_min_pav) = [];
                
                if ~isempty(ampl)
                    arm_swing.amplitude = ampl;
                    arm_swing.pk_ang_vel = pk_ang_vel;
                    arm_swing.start_idx = start_idx;
                    arm_swing.end_idx = end_idx;
                else
                    [arm_swing] = parameter_nan();                   
                end
                
                if ~isnan(start_idx)
                    % Regularity
                    [acf_max] = auto_cor_wrist(angle_pca(start_idx(1):end_idx(end),1),fs);
                    regularity_angle = nanmean(acf_max);
                    [acf_max_av] = auto_cor_wrist(ang_vel_pca(start_idx(1):end_idx(end),1),fs);
                    regularity_ang_vel = nanmean(acf_max_av);
                    
                    arm_swing.regularity_angle = regularity_angle;
                    arm_swing.regularity_ang_vel = regularity_ang_vel;
                    
                    % extract forward and backwards swings seperately
                    swings_neg_pks = ismember(start_idx,idx_neg_pks);
                    ang_vel_forward = pk_ang_vel(swings_neg_pks==1);
                    ang_vel_backward = pk_ang_vel(swings_neg_pks==0);
                    
                    % calculate average peak angular velocity of forward and backward
                    % swings
                    avg_pk_vel_forward = nanmean(ang_vel_forward);
                    avg_pk_vel_backward = nanmean(ang_vel_backward);
                    arm_swing.pk_vel_forward = avg_pk_vel_forward;
                    arm_swing.pk_vel_backward = avg_pk_vel_backward;
                    
                    % calculate percentage swing time (the percentage of the walk that
                    % there was arm swing measured)
                    single_swing_time = nan(size(start_idx));
                    if length(start_idx)>1
                        total_swing_phase  = end_idx(end) - start_idx(1);
                        for i_sp = 1: length(start_idx)
                            single_swing_time(i_sp,1) = end_idx(i_sp) - start_idx(i_sp);
                        end
                        total_swing_time = sum(single_swing_time);
                        perc_time_swing = total_swing_time/total_swing_phase*100;
                        arm_swing.perc_time_swing = round(perc_time_swing);
                    else
                        arm_swing.perc_time_swing = nan;
                    end
                    
                    % frequency per second from the first detected swing till the last
                    % detected swing
                    start_swing = find(idx_remove==0, 1, 'first');
                    end_swing = find(idx_remove==0, 1, 'last')-1;
                    if length(swing_freq) >2 && end_swing- start_swing >2
                        tf = (1:length(swing_freq))* ((steps/window)*(window/fs)) +(steps/fs);
                        freq = interp1(tf(start_swing:end_swing)',swing_freq(start_swing:end_swing,1),  start_swing+1: round(end_swing* ((steps/window)*(window/fs)) +(steps/fs)))';
                        freq = [nan; freq];
                        
                        arm_swing.frequency = freq; % per second
                    else
                        arm_swing.frequency = nan;
                    end
                end
                clearvars -except arm_swing ang_vel_deg fs TH_min_ampl nr_imu i_imu arm_swing_l ang_vel_pca ang_vel_ts_l varargin
            else
                [arm_swing] = parameter_nan();
            end
        else
            warning('no periodical pattern detected')
            [arm_swing] = parameter_nan();
        end
    else
        warning('data too short to calculate arm swing parameters')
        [arm_swing] = parameter_nan();
    end
    if nr_imu == 2 && i_imu == 1
        arm_swing_l = arm_swing;
        ang_vel_ts_l = ang_vel_pca(:,1);
    elseif nr_imu == 2 && i_imu == 2
        arm_swing_r = arm_swing;
        ang_vel_ts_r = ang_vel_pca(:,1);
    end
    
end

%% asymmetry and coordination when both arms are measured
% only calculate these parameters when most of the time swings are detected
% in both arms
clear arm_swing
if nr_imu == 2
    
    if length(arm_swing_r.amplitude)<3 || length(arm_swing_l.amplitude)<3
        perc_both_swing = nan;
        ampl_asym = nan;
        peak_vel_asym = nan;
        cross_cor_max = nan;
    else
        
        % include swing if there is a swing in the other arm within .5 second before
        % and after the swing
        both_swing_l = zeros(size(arm_swing_l.start_idx));
        both_swing_r = zeros(size(arm_swing_r.start_idx));
        time_both_swing_l = zeros(size(arm_swing_l.start_idx));
        time_both_swing_r = zeros(size(arm_swing_r.start_idx));
        for i_as = 1:length(arm_swing_l.start_idx)
            rng= (arm_swing_l.start_idx(i_as)-(round(fs*.5)): arm_swing_l.start_idx(i_as)+(round(fs*.5)))';
            sim_r = ismember(rng,arm_swing_r.start_idx);
            if sum(sim_r)>=1
                both_swing_l(i_as,1) = 1;
                time_both_swing_l(i_as,1) = arm_swing_l.end_idx(i_as) - arm_swing_l.start_idx(i_as);
            end
        end
        perc_both_swing_l = ceil(sum(time_both_swing_l)/(arm_swing_l.end_idx(end)- arm_swing_l.start_idx(1)) * 100);
        
        for i_as = 1:length(arm_swing_r.start_idx)
            rng= (arm_swing_r.start_idx(i_as)-(round(fs*.5)): arm_swing_r.start_idx(i_as)+(round(fs*.5)))';
            sim_l = ismember(rng,arm_swing_l.start_idx);
            if sum(sim_l)>=1
                both_swing_r(i_as,1) = 1;
                time_both_swing_r(i_as,1) = arm_swing_r.end_idx(i_as) - arm_swing_r.start_idx(i_as);
            end
        end
        perc_both_swing_r = ceil(sum(time_both_swing_r)/(arm_swing_r.end_idx(end)- arm_swing_r.start_idx(1)) * 100);
        perc_both_swing = min([perc_both_swing_l , perc_both_swing_r]);
        
        
        % no parameters calculated when less than 3 swings are present or
        % when less than 60% of the time there is no arm swing in both arms
        if sum(both_swing_l)<3 || sum(both_swing_r)<3 || perc_both_swing <60
            ampl_asym = nan;
            peak_vel_asym = nan;
            cross_cor_max = nan;
        else
            
            ampl_l_asym = arm_swing_l.amplitude(both_swing_l==1);
            peak_vel_l_asym = arm_swing_l.pk_ang_vel(both_swing_l==1);
            ampl_r_asym = arm_swing_r.amplitude(both_swing_r==1);
            peak_vel_r_asym = arm_swing_r.pk_ang_vel(both_swing_r==1);
            
            % asymmetry index
            avg_ampl_l = nanmean(ampl_l_asym);
            avg_ampl_r = nanmean(ampl_r_asym);
            avg_peak_vel_l = nanmean(peak_vel_l_asym);
            avg_peak_vel_r = nanmean(peak_vel_r_asym);
            ampl_asym = (avg_ampl_l - avg_ampl_r)/(max([avg_ampl_r , avg_ampl_l]))*100; % as in Killeen 2018
            peak_vel_asym = (avg_peak_vel_l - avg_peak_vel_r)/(max([avg_peak_vel_r , avg_peak_vel_l]))*100;
            
            
            % coordination index
            %cross-correlation (normalized)
            start_idx_l = arm_swing_l.start_idx(both_swing_l==1); 
            start_idx_r = arm_swing_r.start_idx(both_swing_r==1); 
            if length(start_idx_l) >= length(start_idx_r)
                start_swings = start_idx_l;
            else
                start_swings = start_idx_r;
            end
            
            cross_cor_max_cc = nan(length(start_swings),1);
            for i_cc = 1:length(start_swings)
                if start_swings(i_cc)> round(.5*fs) && start_swings(i_cc) < length(ang_vel_ts_l)-round(.5*fs)
                    cross_cor = xcorr(ang_vel_ts_l(start_swings(i_cc)-round(.5*fs): start_swings(i_cc)+round(.5*fs)),...
                        ang_vel_ts_r(start_swings(i_cc)-round(.5*fs): start_swings(i_cc)+round(.5*fs)),'coeff');
                    cross_cor_max_cc(i_cc,1) = abs(min(cross_cor));
                else
                    cross_cor_max_cc(i_cc,1) = nan;
                end
            end
            cross_cor_max = nanmean(cross_cor_max_cc);
        end
        
        
    end
    
    
    arm_swing.arm_swing_l = arm_swing_l;
    arm_swing.arm_swing_r = arm_swing_r;
    arm_swing.perc_both_swing = perc_both_swing;
    arm_swing.amplitude_asymmetry = ampl_asym;
    arm_swing.peak_velocity_asymmetry = peak_vel_asym;
    arm_swing.coordination_max = cross_cor_max;
    
    
else
    arm_swing.arm_swing_l = arm_swing_l;
    arm_swing.arm_swing_r = arm_swing_r;
    arm_swing.perc_both_swing = nan;
    arm_swing.amplitude_asymmetry = nan;
    arm_swing.peak_velocity_asymmetry = nan;
    arm_swing.coordination_max = nan;
    
    
end


%% subfunction

    function [acf_max] = auto_cor_wrist(gyro,f_s)
        % auto_cor_wrist caluclated the maximum autocorrelation of
        % gyroscope data from the wrist
        
        % input: gyro = gyroscope data (Nx1)
        %       f_s = sample frequency
        % output: acf_max = maximum autocorrelation
        
        % Gerhard Schmidt; Kiel University
        
        %**************************************************************************
        % Basic parameters
        %**************************************************************************
        r_in_ms               =   20;
        N_win_acf_in_ms       = 4500;
        N_acf_in_ms           = 2500;
        acf_start_index_in_ms =  300;
        
        r               = round(r_in_ms/1000*f_s);
        N_win_acf       = round(N_win_acf_in_ms/1000*f_s);
        N_acf           = round(N_acf_in_ms/1000*f_s);
        N_sig           = length(gyro);
        acf_start_index = round(acf_start_index_in_ms/1000*f_s);
        
        %**************************************************************************
        % Extend the signal
        %**************************************************************************
        gyro_ext = [zeros(round(N_win_acf/2),1); gyro; zeros(round(N_win_acf/2),1)];
        
        %**************************************************************************
        % Autocorrelation analysis
        %**************************************************************************
        acf_mat          = zeros(N_acf+1,ceil(N_sig/r));
        h_win            = tukeywin(N_win_acf,0.3);
        acf_max          = nan(ceil(N_sig/r),1);
        
        
        k_acf = 0;
        for k = 1:r:length(gyro_ext)-N_win_acf
            
            %**********************************************************************
            % Increment frame index
            %**********************************************************************
            k_acf = k_acf + 1;
            
            %**********************************************************************
            % Extract windowed signal frame
            %**********************************************************************
            sig_win = gyro_ext(k:k+N_win_acf-1);
            sig_win = sig_win .* h_win;
            
            %**********************************************************************
            % Estimate autocorrelation
            %**********************************************************************
            acf_curr     = xcorr(sig_win,N_acf,'unbiased');
            acf_curr     = acf_curr(N_acf+1:end);
            acf_curr     = acf_curr / (acf_curr(1) + eps);
            acf_mat(:,k_acf) = acf_curr;
            
            %**********************************************************************
            % Maximum autocorrelation
            %**********************************************************************
            [acf_max(k_acf,1)] = max(acf_curr(acf_start_index:end));
        end
    end

    function [arm_swing] = parameter_nan()
        arm_swing.amplitude = nan;
        arm_swing.pk_ang_vel = nan;
        arm_swing.start_idx = nan;
        arm_swing.end_idx = nan;
        arm_swing.perc_time_swing = nan;
        arm_swing.regularity_angle = nan;
        arm_swing.regularity_ang_vel = nan;
        arm_swing.pk_vel_forward = nan;
        arm_swing.pk_vel_backward = nan;
        arm_swing.frequency = nan;
    end
end

