(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d h i e)
(:init 
(handempty)
(ontable d)
(ontable h)
(ontable i)
(ontable e)
(clear d)
(clear h)
(clear i)
(clear e)
)
(:goal
(and
(on d h)
(on h i)
(on i e)
)))