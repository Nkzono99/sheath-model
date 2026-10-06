! SPDX-License-Identifier: MIT
!> Caller-owned dimensionless root tables. Every query still solves the original equations.
module sheath_model_atlas
  use sheath_model_constants, only: dp
  use sheath_model_search, only: sheath_continuation_options, valid_continuation_options
  use sheath_model_numerics, only: solve_guarded_linear_system
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sheath_equilibrium_atlas, sheath_field_atlas, sheath_atlas_options, sheath_atlas_point

  type :: sheath_atlas_options
    integer :: neighbors = 8
    real(dp) :: max_distance = 1.0_dp
    logical :: interpolate = .true.
  end type

  type :: sheath_atlas_point
    character(len=1) :: branch = ' '
    integer :: component = 0
    ! log(Te/Tscale), asinh(ue/vthe), log(ui/vscale), log(mi/me),
    ! log1p(pressure_factor*Ti/Tscale), log1p(2 sqrt(pi) Gamma_pe/(ni vscale)).
    ! Prescribed-field maps append asinh(E_H / field_scale) as a seventh coordinate.
    real(dp), allocatable :: key(:)
    real(dp) :: coordinates(3) = 0.0_dp
    real(dp), allocatable :: spectrum_shape(:)
  end type

  type, abstract :: sheath_root_atlas
    type(sheath_atlas_options) :: options
    type(sheath_continuation_options) :: continuation
    type(sheath_atlas_point), allocatable, private :: points(:)
  contains
    procedure :: size => atlas_size
    procedure :: point => atlas_point_at
    procedure :: clear => atlas_clear
    procedure :: valid => atlas_valid
    procedure :: neighbors => atlas_neighbors
    procedure :: predictions => atlas_predictions
    procedure :: insert => atlas_insert
    procedure :: write => atlas_write
    procedure :: read => atlas_read
    procedure(atlas_dimension_interface), deferred, private :: dimension
    procedure(atlas_magic_interface), deferred, private :: magic
  end type

  type, extends(sheath_root_atlas) :: sheath_equilibrium_atlas
  contains
    procedure, private :: dimension => equilibrium_dimension
    procedure, private :: magic => equilibrium_magic
  end type

  type, extends(sheath_root_atlas) :: sheath_field_atlas
  contains
    procedure, private :: dimension => field_dimension
    procedure, private :: magic => field_magic
  end type

  abstract interface
    pure integer function atlas_dimension_interface(self) result(n)
      import sheath_root_atlas
      class(sheath_root_atlas), intent(in) :: self
    end function
    pure function atlas_magic_interface(self) result(magic)
      import sheath_root_atlas
      class(sheath_root_atlas), intent(in) :: self
      character(len=32) :: magic
    end function
  end interface

contains

  pure integer function equilibrium_dimension(self) result(n)
    class(sheath_equilibrium_atlas), intent(in) :: self
    n = 6
  end function

  pure integer function field_dimension(self) result(n)
    class(sheath_field_atlas), intent(in) :: self
    n = 7
  end function

  pure function equilibrium_magic(self) result(magic)
    class(sheath_equilibrium_atlas), intent(in) :: self
    character(len=32) :: magic
    magic = 'SHEATH_EQUILIBRIUM_ATLAS'
  end function

  pure function field_magic(self) result(magic)
    class(sheath_field_atlas), intent(in) :: self
    character(len=32) :: magic
    magic = 'SHEATH_FIELD_ATLAS'
  end function

  integer function atlas_size(self) result(count)
    class(sheath_root_atlas), intent(in) :: self
    count = 0
    if (allocated(self%points)) count = size(self%points)
  end function

  function atlas_point_at(self, index) result(point)
    class(sheath_root_atlas), intent(in) :: self
    integer, intent(in) :: index
    type(sheath_atlas_point) :: point
    point = self%points(index)
  end function

  subroutine atlas_clear(self)
    class(sheath_root_atlas), intent(inout) :: self
    if (allocated(self%points)) deallocate (self%points)
  end subroutine

  logical function atlas_valid(self) result(valid)
    class(sheath_root_atlas), intent(in) :: self
    valid = self%options%neighbors > 0 .and. ieee_is_finite(self%options%max_distance) .and. &
        self%options%max_distance > 0.0_dp .and. valid_continuation_options(self%continuation)
  end function

  logical function same_shape(a, b) result(same)
    real(dp), intent(in) :: a(:), b(:)
    same = .false.
    if (size(a) /= size(b)) return
    same = all(abs(a - b) <= 1e-12_dp*max(1.0_dp, abs(a), abs(b)))
  end function

  subroutine atlas_neighbors(self, key, shape, branch, indices)
    class(sheath_root_atlas), intent(in) :: self
    real(dp), intent(in) :: key(:), shape(:)
    character(len=1), intent(in) :: branch
    integer, allocatable, intent(out) :: indices(:)
    integer, allocatable :: order(:)
    real(dp), allocatable :: distances(:)
    real(dp) :: distance
    integer :: i, j, count
    if (.not. self%valid() .or. size(key) /= self%dimension() .or. .not. all(ieee_is_finite(key))) then
      allocate (indices(0))
      return
    end if
    allocate (order(self%size()), distances(self%size()))
    count = 0
    do i = 1, self%size()
      if (self%points(i)%branch /= branch) cycle
      if (.not. same_shape(self%points(i)%spectrum_shape, shape)) cycle
      distance = sqrt(sum((self%points(i)%key - key)**2))
      if (distance > self%options%max_distance) cycle
      j = count
      do while (j > 0)
        if (distances(j) <= distance) exit
        order(j + 1) = order(j)
        distances(j + 1) = distances(j)
        j = j - 1
      end do
      order(j + 1) = i
      distances(j + 1) = distance
      count = count + 1
    end do
    indices = order(:min(count, self%options%neighbors))
  end subroutine

  subroutine atlas_predictions(self, key, shape, branch, values)
    class(sheath_root_atlas), intent(in) :: self
    real(dp), intent(in) :: key(:), shape(:)
    character(len=1), intent(in) :: branch
    real(dp), allocatable, intent(out) :: values(:, :)
    integer, allocatable :: indices(:), components(:)
    real(dp), allocatable :: guesses(:, :), normal(:, :), rhs(:, :), slopes(:, :), offset(:)
    real(dp) :: prediction(3)
    integer :: i, j, c, index, anchor, count, n
    logical :: valid
    call self%neighbors(key, shape, branch, indices)
    n = self%dimension()
    allocate (normal(n, n), rhs(n, 3), slopes(n, 3), offset(n))
    allocate (components(size(indices)))
    allocate (guesses(3, 2*size(indices)))
    count = 0
    components = 0
    do i = 1, size(indices)
      anchor = indices(i)
      c = self%points(anchor)%component
      if (any(components == c)) cycle
      components(i) = c
      normal = 0.0_dp
      rhs = 0.0_dp
      do j = i + 1, size(indices)
        index = indices(j)
        if (self%points(index)%component /= c) cycle
        offset = self%points(index)%key - self%points(anchor)%key
        normal = normal + spread(offset, 2, n)*spread(offset, 1, n)
        rhs = rhs + spread(offset, 2, 3)*spread(self%points(index)%coordinates - self%points(anchor)%coordinates, 1, n)
      end do
      if (self%options%interpolate) then
        do j = 1, n
          normal(j, j) = normal(j, j) + 1e-10_dp
        end do
        slopes = 0.0_dp
        do j = 1, 3
          call solve_guarded_linear_system(n, normal, rhs(:, j), slopes(:, j), valid)
          if (.not. valid) exit
        end do
        if (valid) then
          prediction = self%points(anchor)%coordinates + matmul(key - self%points(anchor)%key, slopes)
          call append_prediction(prediction)
        end if
      end if
      do j = i, size(indices)
        index = indices(j)
        if (self%points(index)%component == c) call append_prediction(self%points(index)%coordinates)
      end do
    end do
    values = guesses(:, :count)
  contains
    subroutine append_prediction(value)
      real(dp), intent(in) :: value(3)
      integer :: other
      if (.not. all(ieee_is_finite(value))) return
      do other = 1, count
        if (sqrt(sum((guesses(:, other) - value)**2)) < 1e-10_dp) return
      end do
      count = count + 1
      guesses(:, count) = value
    end subroutine
  end subroutine

  !> Insert a seed; successful solves, not the table, establish its physical validity.
  subroutine atlas_insert(self, key, shape, branch, coordinates, component)
    class(sheath_root_atlas), intent(inout) :: self
    real(dp), intent(in) :: key(:), shape(:), coordinates(3)
    character(len=1), intent(in) :: branch
    integer, intent(in), optional :: component
    type(sheath_atlas_point) :: point
    integer, allocatable :: indices(:)
    integer :: i, j, c, max_component
    real(dp) :: distance, best_distance
    logical :: occupied
    if (.not. self%valid() .or. size(key) /= self%dimension()) return
    if (.not. all(ieee_is_finite(key)) .or. .not. all(ieee_is_finite(coordinates))) return
    if (.not. all(ieee_is_finite(shape)) .or. index('ABC', branch) == 0 .or. any(key(5:6) < 0.0_dp)) return
    if (branch /= 'A' .and. coordinates(3) /= 0.0_dp) return
    if (present(component)) then
      if (component < 1) return
    end if
    c = 0
    max_component = 0
    if (.not. allocated(self%points)) allocate (self%points(0))
    call self%neighbors(key, shape, branch, indices)
    do i = 1, self%size()
      max_component = max(max_component, self%points(i)%component)
      if (self%points(i)%branch /= branch) cycle
      if (.not. same_shape(self%points(i)%spectrum_shape, shape)) cycle
      if (sqrt(sum((key - self%points(i)%key)**2)) < 1e-12_dp .and. &
          sqrt(sum((coordinates - self%points(i)%coordinates)**2)) < 1e-6_dp) return
    end do
    if (present(component)) c = component
    if (.not. present(component)) then
      best_distance = huge(1.0_dp)
      do i = 1, size(indices)
        distance = sqrt(sum((coordinates - self%points(indices(i))%coordinates)**2))
        if (distance > self%continuation%max_root_distance .or. distance >= best_distance) cycle
        occupied = .false.
        do j = 1, self%size()
          if (self%points(j)%component /= self%points(indices(i))%component .or. self%points(j)%branch /= branch) cycle
          if (sqrt(sum((key - self%points(j)%key)**2)) < 1e-12_dp) occupied = .true.
        end do
        if (occupied) cycle
        c = self%points(indices(i))%component
        best_distance = distance
      end do
      if (c == 0) c = max_component + 1
    end if
    do i = 1, self%size()
      if (self%points(i)%branch /= branch .or. self%points(i)%component /= c) cycle
      if (.not. same_shape(self%points(i)%spectrum_shape, shape)) cycle
      if (sqrt(sum((key - self%points(i)%key)**2)) < 1e-12_dp .and. &
          sqrt(sum((coordinates - self%points(i)%coordinates)**2)) < 1e-6_dp) return
    end do
    point%branch = branch
    point%component = c
    point%key = key
    point%coordinates = coordinates
    point%spectrum_shape = shape
    self%points = [self%points, point]
  end subroutine

  !> Versioned text format; equilibrium maps share their format with Python.
  !! Field maps have their own header and seventh coordinate. Open unit formatted.
  subroutine atlas_write(self, unit, iostat)
    class(sheath_root_atlas), intent(in) :: self
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    integer :: i
    write (unit, '(a,1x,i0,1x,i0)', iostat=iostat) trim(self%magic()), 1, self%size()
    if (iostat /= 0) return
    do i = 1, self%size()
      write (unit, '(a,1x,i0,1x,i0,1x,*(es25.17,1x))', iostat=iostat) self%points(i)%branch, &
          self%points(i)%component, size(self%points(i)%spectrum_shape), self%points(i)%key, self%points(i)%coordinates
      if (iostat /= 0) return
      if (size(self%points(i)%spectrum_shape) > 0) then
        write (unit, '(*(es25.17,1x))', iostat=iostat) self%points(i)%spectrum_shape
        if (iostat /= 0) return
      end if
    end do
  end subroutine

  !> Read transactionally. Malformed data leaves the existing table unchanged.
  subroutine atlas_read(self, unit, iostat)
    class(sheath_root_atlas), intent(inout) :: self
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    type(sheath_atlas_point), allocatable :: points(:)
    character(len=32) :: magic
    integer :: count, version, shape_count, i
    read (unit, *, iostat=iostat) magic, version, count
    if (iostat /= 0) return
    iostat = 1
    if (magic /= self%magic() .or. version /= 1 .or. count < 0) return
    allocate (points(count))
    do i = 1, count
      allocate (points(i)%key(self%dimension()))
      read (unit, *, iostat=iostat) points(i)%branch, points(i)%component, shape_count, points(i)%key, points(i)%coordinates
      if (iostat /= 0) return
      iostat = 1
      if (shape_count < 0 .or. points(i)%component < 1 .or. index('ABC', points(i)%branch) == 0) return
      if (.not. all(ieee_is_finite(points(i)%key)) .or. .not. all(ieee_is_finite(points(i)%coordinates))) return
      if (any(points(i)%key(5:6) < 0.0_dp)) return
      if (points(i)%branch /= 'A' .and. points(i)%coordinates(3) /= 0.0_dp) return
      allocate (points(i)%spectrum_shape(shape_count))
      if (shape_count > 0) then
        read (unit, *, iostat=iostat) points(i)%spectrum_shape
        if (iostat /= 0) return
        iostat = 1
        if (.not. all(ieee_is_finite(points(i)%spectrum_shape))) return
      end if
    end do
    self%points = points
    iostat = 0
  end subroutine
end module sheath_model_atlas
