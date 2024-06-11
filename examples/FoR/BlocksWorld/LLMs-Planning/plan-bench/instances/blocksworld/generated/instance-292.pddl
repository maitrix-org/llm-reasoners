(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h f g l)
(:init 
(handempty)
(ontable h)
(ontable f)
(ontable g)
(ontable l)
(clear h)
(clear f)
(clear g)
(clear l)
)
(:goal
(and
(on h f)
(on f g)
(on g l)
)))