(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g i f)
(:init 
(handempty)
(ontable g)
(ontable i)
(ontable f)
(clear g)
(clear i)
(clear f)
)
(:goal
(and
(on g i)
(on i f)
)))