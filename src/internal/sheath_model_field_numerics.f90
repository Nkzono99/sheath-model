! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao の初期推定と、汎用 Newton 法へ渡す物理残差の接続。
!! 数学的な反復処理は sheath_model_numerics、残差の定義と許容条件は physics が担当する。
submodule(sheath_model_field) sheath_model_field_numerics
  use sheath_model_numerics, only: solve_guarded_system
  implicit none

contains

  module subroutine solve_field_branch( &
      params, branch, target_field_hat, &
      y0, y_out, &
      final_norm, iterations, &
      success, evaluations, lm_steps &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat, y0(3)
    real(dp), intent(out) :: y_out(3), final_norm
    integer, intent(out) :: iterations, evaluations, lm_steps
    logical, intent(out) :: success

    integer :: n

    n = merge(3, 2, branch == 'A')
    y_out = y0
    call solve_guarded_system(n, y0(1:n), field_residual, params%search, &
        y_out(1:n), final_norm, iterations, success, evaluations, lm_steps)

  contains

    subroutine field_residual(y, f, valid)
      real(dp), intent(in) :: y(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
      real(dp) :: encoded(3), residual(3)

      encoded = y0
      encoded(1:n) = y
      call evaluate_charge_residual(params, branch, target_field_hat, encoded, residual, valid)
      f = residual(1:n)
    end subroutine field_residual
  end subroutine solve_field_branch

end submodule sheath_model_field_numerics
