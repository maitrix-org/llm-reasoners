

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b e)
(ontable c)
(ontable d)
(on e c)
(clear a)
(clear b)
(clear d)
)
(:goal
(and
(on a e)
(on d c)
(on e b))
)
)


