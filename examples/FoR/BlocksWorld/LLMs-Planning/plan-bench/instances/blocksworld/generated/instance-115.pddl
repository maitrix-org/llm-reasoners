(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j h e)
(:init 
(handempty)
(ontable j)
(ontable h)
(ontable e)
(clear j)
(clear h)
(clear e)
)
(:goal
(and
(on j h)
(on h e)
)))