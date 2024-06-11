(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f j e)
(:init 
(handempty)
(ontable f)
(ontable j)
(ontable e)
(clear f)
(clear j)
(clear e)
)
(:goal
(and
(on f j)
(on j e)
)))