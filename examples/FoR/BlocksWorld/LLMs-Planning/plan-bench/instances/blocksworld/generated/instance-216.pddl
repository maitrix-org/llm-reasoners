(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e a f g)
(:init 
(handempty)
(ontable e)
(ontable a)
(ontable f)
(ontable g)
(clear e)
(clear a)
(clear f)
(clear g)
)
(:goal
(and
(on e a)
(on a f)
(on f g)
)))