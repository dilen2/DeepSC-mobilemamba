# Modified by Mzero #20240123
# Copyright (C) 2023, Tri Dao, Albert Gu.

import math
import torch
import torch.nn.functional as F
import pytest
import logging
import time
from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange, repeat


def build_selective_scan_fn(selective_scan_cuda: object = None, mode="mamba_ssm"):
    """构建选择性扫描函数
    Args:
        selective_scan_cuda: CUDA实现的选择性扫描模块
        mode: 运行模式 ("mamba_ssm", "sscore", "sstest")
    Returns:
        选择性扫描函数
    """
    MODE = mode

    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, nrows=1, backnrows=-1):
            """前向传播
            Args:
                u: 输入张量
                delta: 时间步长
                A: 状态转移矩阵
                B: 输入投影矩阵
                C: 输出投影矩阵
                D: 跳跃连接矩阵
                z: 门控信号
                delta_bias: 时间步长偏置
                delta_softplus: 是否使用softplus激活
                return_last_state: 是否返回最后状态
                nrows: 行数
                backnrows: 反向传播行数
            Returns:
                输出张量和可选的最后状态
            """
            # 确保张量连续
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()

            # 处理B和C的维度
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True

            # 处理数据类型
            if D is not None and (D.dtype != torch.float):
                ctx._d_dtype = D.dtype
                D = D.float()
            if delta_bias is not None and (delta_bias.dtype != torch.float):
                ctx._delta_bias_dtype = delta_bias.dtype
                delta_bias = delta_bias.float()

            # 验证输入维度
            assert u.shape[1] % (B.shape[1] * nrows) == 0 
            assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile

            if backnrows > 0:
                assert u.shape[1] % (B.shape[1] * backnrows) == 0 
                assert backnrows in [1, 2, 3, 4] # 8+ is too slow to compile
            else:
                backnrows = nrows
            ctx.backnrows = backnrows
            
            # 执行前向传播
            if MODE in ["mamba_ssm"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            elif MODE in ["sscore"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            elif MODE in ["sstest"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, nrows)
            else:
                raise NotImplementedError(f"不支持的模式: {MODE}")

            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None

            last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                return out if not return_last_state else (out, last_state)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                if MODE in ["mamba_ssm", "sstest"]:
                    out_z = rest[0]
                    return out_z if not return_last_state else (out_z, last_state)
                elif MODE in ["sscore"]:
                    return out if not return_last_state else (out, last_state)

        @staticmethod
        def backward(ctx, dout, *args):
            """反向传播
            Args:
                dout: 输出梯度
                *args: 其他参数
            Returns:
                各参数的梯度
            """
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors

            if dout.stride(-1) != 1:
                dout = dout.contiguous()

            # 执行反向传播
            if MODE in ["mamba_ssm"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                    False # option to recompute out_z, not used here
                )
            elif MODE in ["sstest"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                    False, ctx.backnrows  # option to recompute out_z, not used here
                )
            elif MODE in ["sscore"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.backnrows
                )
            else:
                raise NotImplementedError(f"不支持的模式: {MODE}")
            
            # 处理梯度
            dz = rest[0] if ctx.has_z else None
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            
            _dD = None
            if D is not None:
                if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                    _dD = dD.to(ctx._d_dtype)
                else:
                    _dD = dD

            _ddelta_bias = None
            if delta_bias is not None:
                if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                    _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
                else:
                    _ddelta_bias = ddelta_bias

            return (du, ddelta, dA, dB, dC,
                    dD if D is not None else None,
                    dz,
                    ddelta_bias if delta_bias is not None else None,
                    None, None, None, None)

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, nrows=1, backnrows=-1):
        """选择性扫描函数
        Args:
            u: 输入张量
            delta: 时间步长
            A: 状态转移矩阵
            B: 输入投影矩阵
            C: 输出投影矩阵
            D: 跳跃连接矩阵
            z: 门控信号
            delta_bias: 时间步长偏置
            delta_softplus: 是否使用softplus激活
            return_last_state: 是否返回最后状态
            nrows: 行数
            backnrows: 反向传播行数
        Returns:
            输出张量和可选的最后状态
        """
        return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, nrows, backnrows)

    return selective_scan_fn


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """参考实现的选择性扫描函数
    Args:
        u: 输入张量 (B D L)
        delta: 时间步长 (B D L)
        A: 状态转移矩阵 (D N) 或 (D N)
        B: 输入投影矩阵 (D N) 或 (B N L) 或 (B N 2L) 或 (B G N L)
        C: 输出投影矩阵 (D N) 或 (B N L) 或 (B N 2L) 或 (B G N L)
        D: 跳跃连接矩阵 (D)
        z: 门控信号 (B D L)
        delta_bias: 时间步长偏置 (D), fp32
        delta_softplus: 是否使用softplus激活
        return_last_state: 是否返回最后状态
    Returns:
        输出张量 (B D L) 和可选的最后状态 (B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# 设置运行模式
MODE = "mamba_ssm_sscore" # 1344 items pass

# 根据模式导入相应的模块
if MODE in ["mamba_ssm"]:
    import selective_scan_cuda as selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["sscore"]:
    import selective_scan_cuda_core
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_core, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["sstest"]:
    import selective_scan_cuda_test
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_test, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["mamba_ssm_sscore"]:
    import selective_scan_cuda_core
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_core, mode="sscore")
    selective_scan_ref = build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm")
elif MODE in ["mamba_ssm_sstest"]:
    import selective_scan_cuda_test
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_test, mode="sstest")
    selective_scan_ref = build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm")
else:
    raise NotImplementedError(f"不支持的模式: {MODE}")

logging.info(f"使用模式: {MODE}")
time.sleep(10)  # 等待CUDA初始化


@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('delta_softplus', [False, True])
@pytest.mark.parametrize('has_z', [False])
@pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("nrows", [1, 2, 3, 4])
def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seqlen, itype, wtype, nrows):
    """测试选择性扫描函数
    Args:
        is_variable_B: B是否为变量
        is_variable_C: C是否为变量
        varBC_groups: BC变量组数
        has_D: 是否有D
        has_z: 是否有z
        has_delta_bias: 是否有delta_bias
        delta_softplus: 是否使用softplus
        return_last_state: 是否返回最后状态
        seqlen: 序列长度
        itype: 输入类型
        wtype: 权重类型
        nrows: 行数
    """
    logging.info(f"测试配置: seqlen={seqlen}, itype={itype}, wtype={wtype}, nrows={nrows}")
    
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip("此配置不适用")

    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)

    # 设置随机种子
    torch.random.manual_seed(0)
    
    # 初始化参数
    batch_size = 2
    dim = 24
    dstate = 8
    is_complex = wtype == torch.complex64
    
    # 创建输入张量
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    
    # 创建B张量
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                    requires_grad=True)
    
    # 创建C张量
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                    requires_grad=True)
    
    # 创建其他张量
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True) if has_D else None
    z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True) if has_z else None
    delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_() if has_delta_bias else None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
    
    # 创建参考张量
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    
    # 执行前向传播
    out, *rest = selective_scan_fn(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=return_last_state, nrows=nrows
    )
    if return_last_state:
        state = rest[0]
    
    out_ref, *rest = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
    if return_last_state:
        state_ref = rest[0]
    
    # 检查输出
    logging.info(f'输出最大差异: {(out - out_ref).abs().max().item()}')
    logging.info(f'输出平均差异: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        logging.info(f'状态最大差异: {(state - state_ref).abs().max().item()}')
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    
    # 执行反向传播
    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)
    
    # 检查梯度
    logging.info(f'du最大差异: {(u.grad - u_ref.grad).abs().max().item()}')
    logging.info(f'ddelta最大差异: {(delta.grad - delta_ref.grad).abs().max().item()}')
    logging.info(f'dA最大差异: {(A.grad - A_ref.grad).abs().max().item()}')
    logging.info(f'dB最大差异: {(B.grad - B_ref.grad).abs().max().item()}')
    logging.info(f'dC最大差异: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        logging.info(f'dD最大差异: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_z:
        logging.info(f'dz最大差异: {(z.grad - z_ref.grad).abs().max().item()}')
    if has_delta_bias:
        logging.info(f'ddelta_bias最大差异: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')
    
    # 验证梯度
    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                          atol=atolw if not is_variable_B else atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                          atol=atolw if not is_variable_C else atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)



