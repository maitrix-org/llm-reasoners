(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f e k)
(:init 
(handempty)
(ontable f)
(ontable e)
(ontable k)
(clear f)
(clear e)
(clear k)
)
(:goal
(and
(on f e)
(on e k)
)))