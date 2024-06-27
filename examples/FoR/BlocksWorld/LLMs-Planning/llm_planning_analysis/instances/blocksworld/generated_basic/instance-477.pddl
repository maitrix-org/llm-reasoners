

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b d)
(ontable c)
(ontable d)
(on e b)
(clear a)
(clear c)
(clear e)
)
(:goal
(and
(on b d)
(on c e)
(on e a))
)
)


