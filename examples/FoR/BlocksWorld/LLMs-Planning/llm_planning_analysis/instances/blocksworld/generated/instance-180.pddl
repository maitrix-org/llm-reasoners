(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f c i l)
(:init 
(handempty)
(ontable f)
(ontable c)
(ontable i)
(ontable l)
(clear f)
(clear c)
(clear i)
(clear l)
)
(:goal
(and
(on f c)
(on c i)
(on i l)
)))