! SPDX-License-Identifier: MIT
! Generate the numerical data behind the README figures using the public API.
! Usage: fpm run --example readme_data -- OUTPUT_DIRECTORY [NX NY [both|equilibrium|field]]
program readme_data
  use, intrinsic :: iso_fortran_env, only: output_unit
  use sheath_model
  implicit none
  character(len=512) :: directory, argument
  integer :: nx, ny, i, j, unit, kind, first_map, last_map, started, finished, clock_rate
  integer, allocatable :: types(:, :), unresolved(:, :), rejected(:, :), candidates(:, :)
  real(dp), allocatable :: x(:), ratio(:)
  real(dp), parameter :: pi = acos(-1.0_dp), alpha_field = 20.0_dp

  directory = 'docs/figures/data'
  nx = 65
  ny = 33
  first_map = 1
  last_map = 2
  if (command_argument_count() >= 1) call get_command_argument(1, directory)
  if (command_argument_count() >= 3) then
    call get_command_argument(2, argument)
    read (argument, *) nx
    call get_command_argument(3, argument)
    read (argument, *) ny
  end if
  if (command_argument_count() >= 4) then
    call get_command_argument(4, argument)
    select case (trim(argument))
    case ('equilibrium')
      last_map = 1
    case ('field')
      first_map = 2
    case ('both')
    case default
      error stop 'Map selection must be both, equilibrium, or field.'
    end select
  end if
  if (nx < 3 .or. ny < 3) error stop 'Use at least three points on each map axis.'
  allocate (types(nx, ny), unresolved(nx, ny), rejected(nx, ny), candidates(nx, ny), x(nx), ratio(ny))
  open (newunit=unit, file=trim(directory)//'/profile_metadata.csv', status='replace')
  write (unit, '(a)') 'type,alpha_deg,phi_surface_v,phi_min_v,electron_normalization_m3,current_a_m2,residual,turning_height_m'
  call write_profile('A', 60.0_dp)
  call write_profile('B', 20.0_dp)
  call write_profile('C', 10.0_dp)
  close (unit)
  do j = 1, ny
    ratio(j) = 0.5_dp*32.0_dp**(real(j - 1, dp)/real(ny - 1, dp))
  end do
  do kind = first_map, last_map
    call system_clock(started, clock_rate)
    do i = 1, nx
      if (kind == 1) then
        x(i) = 1.0_dp + 88.0_dp*real(i - 1, dp)/real(nx - 1, dp)
      else
        x(i) = -2.0_dp + 4.0_dp*real(i - 1, dp)/real(nx - 1, dp)
      end if
    end do
    do j = 1, ny
      ! Each solve is independent. Without OpenMP this remains a serial example.
      !$omp parallel do default(none) shared(kind, nx, j, x, ratio, types, unresolved, rejected, candidates) private(i)
      do i = 1, nx
        if (kind == 1) then
          call equilibrium_point(x(i), ratio(j), types(i, j), unresolved(i, j), rejected(i, j))
          candidates(i, j) = popcnt(types(i, j))
        else
          call field_point(x(i), ratio(j), types(i, j), unresolved(i, j), rejected(i, j), candidates(i, j))
        end if
      end do
      !$omp end parallel do
      print '(a,i0,a,i0,a,i0)', 'Map ', kind, ': row ', j, '/', ny
      flush (output_unit)
    end do
    if (kind == 1) then
      open (newunit=unit, file=trim(directory)//'/equilibrium_map.csv', status='replace')
    else
      open (newunit=unit, file=trim(directory)//'/field_map.csv', status='replace')
    end if
    write (unit, '(a)') 'x,source_ratio,types_found,unresolved_types,rejected_types,candidate_count'
    do j = 1, ny
      do i = 1, nx
        write (unit, '(es16.8,",",es16.8,4(",",i0))') &
          x(i), ratio(j), types(i, j), unresolved(i, j), rejected(i, j), candidates(i, j)
      end do
    end do
    close (unit)
    call system_clock(finished)
    print '(a,i0,a,f10.2)', 'Map ', kind, ' elapsed seconds: ', real(finished - started, dp)/clock_rate
  end do
contains
  subroutine write_profile(branch, alpha)
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: alpha
    type(zhao_equilibrium_input) :: input
    type(zhao_profile_options) :: options
    type(zhao_profile_result) :: profile
    integer(i32) :: status
    integer :: k, file, turning
    character(len=512) :: message
    input = zhao_equilibrium_input(branch=branch, sun_elevation_deg=alpha, electron_drift_mode='zero')
    options = zhao_profile_options(points_per_segment=8000, max_distance_m=150.0_dp, potential_cutoff_v=2.2e-4_dp)
    call solve_profile(input, options, profile, status, message)
    if (status /= sheath_ok) then
      print *, trim(message)
      error stop 'README representative profile failed.'
    end if
    turning = minloc(abs(profile%z_m - profile%turning_height_m), dim=1)
    open (newunit=file, file=trim(directory)//'/profile_'//branch//'.csv', status='replace')
    write (file, '(a)') 'z_m,potential_v,electric_field_v_m,charge_c_m3'
    do k = 1, size(profile%z_m)
      if (mod(k - 1, 8) /= 0 .and. k /= turning .and. k /= size(profile%z_m)) cycle
      write (file, '(es18.10,3(",",es18.10))') profile%z_m(k), profile%potential_v(k), &
        profile%electric_field_v_m(k), profile%density(k)%charge_c_m3
    end do
    close (file)
    write (unit, '(a,7(",",es18.10))') branch, alpha, profile%equilibrium%surface_potential_v, &
      profile%equilibrium%minimum_potential_v, profile%equilibrium%ambient_electron_density_m3, &
      profile%equilibrium%net_current_a_m2, profile%equilibrium%residual_norm, profile%turning_height_m
    print '(a,a,a,4es16.7)', 'Type ', branch, ' alpha, phi_H, phi_min, J_z: ', alpha, &
      profile%equilibrium%surface_potential_v, profile%equilibrium%minimum_potential_v, &
      profile%equilibrium%net_current_a_m2
  end subroutine write_profile

  subroutine equilibrium_point(alpha, source_ratio, found, unknown, refused)
    real(dp), intent(in) :: alpha, source_ratio
    integer, intent(out) :: found, unknown, refused
    character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
    type(zhao_equilibrium_input) :: input
    type(zhao_equilibrium_result) :: result
    integer(i32) :: status
    integer :: k
    character(len=512) :: message
    input = zhao_equilibrium_input(sun_elevation_deg=alpha, electron_drift_mode='zero')
    input%photoelectron_reference_density_m3 = input%ion_density_m3*source_ratio
    found = 0
    unknown = 0
    refused = 0
    do k = 1, 3
      input%branch = branches(k)
      call solve_equilibrium(input, result, status, message)
      select case (status)
      case (sheath_ok)
        found = ibset(found, k - 1)
      case (sheath_no_physical_solution)
        refused = ibset(refused, k - 1)
      case (sheath_numerical_failure)
        unknown = ibset(unknown, k - 1)
      case default
        error stop 'Unexpected equilibrium map status.'
      end select
    end do
  end subroutine equilibrium_point

  subroutine field_point(field, source_ratio, found, unknown, refused, count)
    real(dp), intent(in) :: field, source_ratio
    integer, intent(out) :: found, unknown, refused, count
    type(zhao_field_input) :: input
    type(zhao_field_result), allocatable :: roots(:)
    integer(i32) :: status
    integer :: k
    character(len=512) :: message
    input = zhao_field_input(electric_field_v_m=field, electron_drift_mps=0.0_dp)
    input%ion_drift_mps = 468.0e3_dp*sin(alpha_field*pi/180.0_dp)
    input%photoelectron_source_density_m3 = input%ion_density_m3*source_ratio*sin(alpha_field*pi/180.0_dp)
    call solve_prescribed_field_candidates(input, roots, status, message)
    found = 0
    unknown = 0
    refused = 0
    count = 0
    select case (status)
    case (sheath_ok)
      count = size(roots)
      do k = 1, count
        select case (roots(k)%branch)
        case ('A')
          found = ibset(found, 0)
        case ('B')
          found = ibset(found, 1)
        case ('C')
          found = ibset(found, 2)
        end select
      end do
    case (sheath_no_physical_solution)
      refused = 7
    case (sheath_numerical_failure)
      unknown = 7
    case default
      error stop 'Unexpected prescribed-field map status.'
    end select
  end subroutine field_point
end program readme_data
