program validation_probe
  use sheath_model
  implicit none
  type(zhao_field_input) :: p
  type(zhao_field_result), allocatable :: roots(:)
  integer(i32) :: status
  integer :: c, j, out, stat
  character(len=512) :: message
  real(dp) :: fields(3)
  fields = [1.4_dp, 1.6_dp, 1.8_dp]
  open(newunit=out,file='validation/field_roots.csv',status='replace')
  open(newunit=stat,file='validation/field_status.csv',status='replace')
  write(out,'(a)') 'case,branch,E_H,ni,npe0,Te,Tpe,ve,vi,phi_H,phi_min,Ne,Gamma_e,Gamma_i,Gamma_escape,J,residual,min_E2'
  write(stat,'(a)') 'case,requested_branch,status,count,message'
  do c=0,11
    p=zhao_field_input()
    p%electron_drift_mps=0.0_dp
    if (c>=1 .and. c<=3) then
      p%photoelectron_source_density_m3=64e6_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
      p%ion_drift_mps=468e3_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
      p%electric_field_v_m=real(c-2,dp)*0.01_dp
    else if (c>=4 .and. c<=9) then
      p%photoelectron_source_density_m3=5.5425625842204072e7_dp
      p%electric_field_v_m=fields(mod(c-4,3)+1)
      if (c>=7) p%electron_drift_mps=405299.88897111727_dp
    else if(c>=10) then
      p%branch='A'
      if(c==11) p%branch='C'
      p%electron_drift_mps=4e5_dp
      p%ion_drift_mps=4e5_dp
      p%ion_density_m3=5e6_dp
      p%electron_temperature_ev=10.0_dp
      p%photoelectron_temperature_ev=2.216510787_dp
      p%photoelectron_source_density_m3=7.6619216e7_dp
      p%electric_field_v_m=2.2014259812892867_dp
      if(c==11) p%electric_field_v_m=-p%electric_field_v_m
    end if
    call solve_prescribed_field_candidates(p,roots,status,message)
    j=0
    if(allocated(roots)) j=size(roots)
    write(stat,'(i0,a,a,a,i0,a,i0,a,a)') c,',',trim(p%branch),',',status,',',j,',',trim(message)
    flush(stat)
    if(status/=sheath_ok) cycle
    do j=1,size(roots)
      write(out,'(i0,a,a,16(a,es25.16e3))') c,',',roots(j)%branch, &
        ',',p%electric_field_v_m,',',p%ion_density_m3,',',p%photoelectron_source_density_m3, &
        ',',p%electron_temperature_ev,',',p%photoelectron_temperature_ev, &
        ',',p%electron_drift_mps,',',p%ion_drift_mps, &
        ',',roots(j)%boundary_potential_v,',',roots(j)%minimum_potential_v, &
        ',',roots(j)%ambient_electron_density_m3, &
        ',',roots(j)%electron_inward_flux_m2_s,',',roots(j)%ion_inward_flux_m2_s, &
        ',',roots(j)%photoelectron_escape_flux_m2_s,',',roots(j)%net_current_a_m2, &
        ',',roots(j)%residual_norm,',',roots(j)%minimum_field_squared_hat
    end do
    flush(out)
  end do
  close(out)
  close(stat)
end program validation_probe
