

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a e)
(on b c)
(ontable c)
(on d a)
(on e b)
(clear d)
)
(:goal
(and
(on d e))
)
)


