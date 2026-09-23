program test_photoelectrons
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  use sheath_model
  use sheath_model_constants, only: qe, pi, electron_mass
  use sheath_model_photoelectrons, only: validate_photoelectrons, photoelectron_density, photoelectron_fluxes, &
      photoelectron_density_integral, photoelectron_sqrt_coefficient
  implicit none
  type(photoelectron_source) :: source, analytic
  real(dp) :: edges(5), flux(4), outward, escape, returning, free, captured, exact_free, exact_captured
  real(dp) :: integral, reference, phi, h, weight, speed, coarse_error, fine_error, gamma, edge, width
  real(dp) :: velocity, kinetic, vdf, reference_free, reference_captured, reference_flux
  real(dp), allocatable :: bins(:), rates(:)
  integer :: i, j, k, n
  integer(i32) :: status
  character(len=256) :: message

  edges = [0.0_dp, 0.3_dp, 1.2_dp, 4.0_dp, 9.0_dp]
  flux = [2.0_dp, 7.0_dp, 3.0_dp, 1.0_dp]*1e12_dp
  source = binned_photoelectrons(edges, flux)
  call validate_photoelectrons(source, status, message)
  call check(status == SHEATH_OK, 'nonuniform source')
  do i = 0, 100
    call photoelectron_fluxes(source, electron_mass, real(i, dp)/10.0_dp, outward, escape, returning)
    call near(escape + returning, sum(flux), 4e-15_dp, 'flux conservation, each returning orbit counted once')
  end do
  call photoelectron_fluxes(source, electron_mass, 1.2_dp, outward, escape, returning)
  call near(escape, sum(flux(3:4)), 1e-15_dp, 'integrated bin flux, not differential flux')

  ! Independent local velocity integration of the transported source VDF.
  ! f(v)=2*(dGamma/dK)/(2e/m), with K=m*v^2/(2e)+phi_H-phi.
  speed = sqrt(2.0_dp*qe/electron_mass)
  do j = 1, 3
    phi = -0.7_dp + (j - 1)*1.35_dp
    n = 200000
    h = speed*sqrt(edges(5) - 2.0_dp + phi)/n
    reference_free = 0.0_dp
    reference_captured = 0.0_dp
    reference_flux = 0.0_dp
    do i = 1, n
      velocity = (i - 0.5_dp)*h
      kinetic = (velocity/speed)**2 + 2.0_dp - phi
      vdf = 0.0_dp
      do k = 1, 4
        if (kinetic >= edges(k) .and. kinetic < edges(k + 1)) then
          vdf = 2.0_dp*flux(k)/((edges(k + 1) - edges(k))*speed**2)
        end if
      end do
      if (kinetic >= 2.7_dp) then
        reference_free = reference_free + h*vdf
        reference_flux = reference_flux + h*velocity*vdf
      else
        reference_captured = reference_captured + 2.0_dp*h*vdf
      end if
    end do
    call photoelectron_density(source, electron_mass, 2.0_dp, -0.7_dp, phi, .true., free, captured)
    call photoelectron_fluxes(source, electron_mass, 2.7_dp, outward, escape, returning)
    call near(free, reference_free, 5e-5_dp, 'passing density from independent local VDF')
    call near(captured, reference_captured, 5e-5_dp, 'both returning legs from independent local VDF')
    call near(escape, reference_flux, 5e-5_dp, 'passing flux conserved along independently integrated orbits')
  end do

  ! Independent composite Simpson integration over potential, including bin turning points.
  do j = 1, 2
    n = 32768
    h = 2.7_dp/n
    reference = 0.0_dp
    do i = 0, n
      phi = -0.7_dp + i*h
      call photoelectron_density(source, electron_mass, 2.0_dp, -0.7_dp, phi, j == 1, free, captured)
      weight = 2.0_dp
      if (mod(i, 2) == 1) weight = 4.0_dp
      if (i == 0 .or. i == n) weight = 1.0_dp
      reference = reference + weight*(free + captured)
    end do
    reference = reference*h/3.0_dp
    integral = photoelectron_density_integral(source, electron_mass, 2.0_dp, -0.7_dp, -0.7_dp, 2.0_dp, j == 1)
    call near(integral, reference, 2e-7_dp, 'analytic PE primitive versus independent potential quadrature')
    call near(photoelectron_density_integral(source, electron_mass, 2.0_dp, -0.7_dp, 2.0_dp, -0.7_dp, j == 1), &
        -integral, 1e-15_dp, 'oriented integral')
  end do

  ! A narrow bin and an adjacent floating-point potential interval would lose most digits in naive primitives.
  edge = 4.0_dp
  width = 1e-10_dp
  source = binned_photoelectrons([edge, edge + width], [1e12_dp])
  speed = sqrt(2.0_dp*qe/electron_mass)
  h = nearest(1.0_dp, 1.0_dp) - 1.0_dp
  integral = photoelectron_density_integral(source, electron_mass, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp + h, .false.)
  call near(integral, 1e12_dp*h/(speed*sqrt(5.0_dp)), 2e-11_dp, 'narrow bin and adjacent potential precision')
  source = binned_photoelectrons([0.0_dp, 1.0_dp], [1e12_dp])
  h = 1e-18_dp
  integral = photoelectron_density_integral(source, electron_mass, 0.0_dp, -1.0_dp, -1.0_dp, -1.0_dp + h, .true.)
  call near(integral, 0.0_dp, 0.0_dp, 'identical representable endpoints')
  h = nearest(-1.0_dp, 1.0_dp) + 1.0_dp
  integral = photoelectron_density_integral(source, electron_mass, 0.0_dp, -1.0_dp, -1.0_dp, -1.0_dp + h, .true.)
  call near(integral, (8.0_dp/3.0_dp)*1e12_dp*h*sqrt(h)/speed, 3e-15_dp, 'adjacent turning-point primitive')

  ! At a bin edge, returning and escaping populations have different limiting differential fluxes.
  source = binned_photoelectrons([0.0_dp, 1.0_dp, 2.0_dp], [3e12_dp, 1e12_dp])
  call near(photoelectron_sqrt_coefficient(source, electron_mass, 1.0_dp), 10e12_dp/speed, 2e-15_dp, 'two-sided edge')

  analytic = maxwellian_photoelectrons(6.4e7_dp, 2.2_dp)
  call photoelectron_fluxes(analytic, electron_mass, 0.0_dp, gamma, escape, returning)
  call photoelectron_density(analytic, electron_mass, 3.0_dp, -0.5_dp, 1.0_dp, .true., exact_free, exact_captured)
  do j = 1, 2
    n = 128*8**(j - 1)
    allocate (bins(n + 1), rates(n))
    do i = 1, n + 1
      bins(i) = real(i - 1, dp)*44.0_dp/n
    end do
    rates = gamma*(exp(-bins(:n)/2.2_dp) - exp(-bins(2:)/2.2_dp))
    source = binned_photoelectrons(bins, rates)
    call photoelectron_density(source, electron_mass, 3.0_dp, -0.5_dp, 1.0_dp, .true., free, captured)
    fine_error = max(abs(free/exact_free - 1.0_dp), abs(captured/exact_captured - 1.0_dp))
    if (j == 1) coarse_error = fine_error
    deallocate (bins, rates)
  end do
  call check(fine_error < 0.1_dp*coarse_error .and. fine_error < 1e-3_dp, 'Maxwellian refinement convergence')

  ! Equal outward flux and equal flux-weighted mean energy do not define the source shape.
  source = binned_photoelectrons([0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp], [1e12_dp, 0.0_dp, 1e12_dp])
  call photoelectron_fluxes(source, electron_mass, 2.0_dp, outward, escape, returning)
  call near(escape, 1e12_dp, 1e-15_dp, 'two-lobed source escapes')
  source = binned_photoelectrons([0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp], [0.0_dp, 2e12_dp, 0.0_dp])
  call photoelectron_fluxes(source, electron_mass, 2.0_dp, outward, escape, returning)
  call near(escape, 0.0_dp, 0.0_dp, 'same moments but different shape remains distinguishable')

  source = binned_photoelectrons([0.0_dp, 0.0_dp], [0.0_dp])
  call validate_photoelectrons(source, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'zero bin width rejected')
  source = binned_photoelectrons([0.0_dp, 1.0_dp], [-1.0_dp])
  call validate_photoelectrons(source, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'negative flux rejected')
  source = binned_photoelectrons([0.0_dp, 1.0_dp], [ieee_value(0.0_dp, ieee_quiet_nan)])
  call validate_photoelectrons(source, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'NaN flux rejected')
  print *, 'Photoelectron spectrum moments and precision checks passed.'
contains
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label
      error stop 1
    end if
  end subroutine
  subroutine near(actual, expected, tolerance, label)
    real(dp), intent(in) :: actual, expected, tolerance
    character(len=*), intent(in) :: label
    if (.not. ieee_is_finite(actual) .or. abs(actual - expected) > tolerance*abs(expected)) then
      print *, actual, expected, tolerance
      call check(.false., label)
    end if
  end subroutine
end program
